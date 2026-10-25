"""
src/detector/anomaly.py
=======================
Unsupervised layer of the detector (SRS FR-2.2) — no ground-truth labels used.

Two IsolationForests:

  AnomalyDetector        trader-level. Scores every (account, window) on the
                         behavioural + peer-relative features. Market context is
                         deliberately excluded so that "the market was volatile"
                         is not mistaken for "this account behaved oddly".
  MarketAnomalyDetector  market-level. Scores each window's market regime
                         (returns, volume spikes, flow imbalance). Its score is
                         passed to the supervised layer as context, not used to
                         flag accounts on its own.

Scores are reported as an empirical percentile of the training-score
distribution (0 = most typical, 1 = more anomalous than everything seen in
training), so a threshold of 1 - contamination is directly interpretable.
Features are robust-scaled (median / IQR) because the behavioural ratios are
heavy-tailed. Explanations use SHAP's exact TreeExplainer for isolation forests.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler

logger = logging.getLogger(__name__)

NON_FEATURE_COLUMNS = {"window_end", "commodity", "trader_id", "window_start", "cv_group"}


def behaviour_columns(X: pd.DataFrame) -> list[str]:
    """Trader behavioural features (no market context, no identifiers)."""
    return [c for c in X.columns
            if c not in NON_FEATURE_COLUMNS and not c.startswith("mkt_")
            and pd.api.types.is_numeric_dtype(X[c])]


def market_columns(X: pd.DataFrame) -> list[str]:
    return [c for c in X.columns if c.startswith("mkt_") and c != "mkt_anomaly_score"]


class _PercentileForest:
    """IsolationForest + robust scaling + percentile calibration."""

    def __init__(self, contamination: float, n_estimators: int, max_samples: Any, random_state: int) -> None:
        self.contamination = contamination
        self.scaler = RobustScaler(quantile_range=(10.0, 90.0))
        self.model = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            max_samples=max_samples,
            random_state=random_state,
            n_jobs=-1,
        )
        self.columns: list[str] = []
        self._train_scores: np.ndarray = np.array([])

    # Percentile calibration uses at most this many training rows: scoring all 2M+ cumulative
    # rows took a minute per retrain, and 300k rows put the 99th percentile within ~0.02 points.
    max_calibration_rows = 300_000

    def fit(self, X: pd.DataFrame, columns: list[str]) -> None:
        self.columns = columns
        Z = self.scaler.fit_transform(X[columns].to_numpy(dtype=float))
        self.model.fit(Z)
        if len(Z) > self.max_calibration_rows:
            rng = np.random.default_rng(0)
            Z = Z[rng.choice(len(Z), self.max_calibration_rows, replace=False)]
        self._train_scores = np.sort(-self.model.score_samples(Z))

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        return self.scaler.transform(X[self.columns].to_numpy(dtype=float))

    def percentile(self, X: pd.DataFrame) -> np.ndarray:
        raw = -self.model.score_samples(self.transform(X))
        return np.searchsorted(self._train_scores, raw, side="right") / max(1, len(self._train_scores))


class AnomalyDetector:
    """Trader-level IsolationForest (persona-agnostic)."""

    def __init__(self, config: dict[str, Any]) -> None:
        iso = config.get("isolation_forest", {})
        self.contamination = float(iso.get("contamination", 0.03))
        self._forest = _PercentileForest(
            self.contamination,
            iso.get("n_estimators", 200),
            iso.get("max_samples", "auto"),
            iso.get("random_state", 42),
        )
        self.is_fitted = False

    @property
    def feature_columns(self) -> list[str]:
        return self._forest.columns

    def fit(self, X: pd.DataFrame) -> None:
        cols = behaviour_columns(X)
        logger.info("Fitting trader-level IsolationForest on %d rows x %d features", len(X), len(cols))
        self._forest.fit(X, cols)
        self.is_fitted = True

    def predict_scores(self, X: pd.DataFrame) -> pd.Series:
        """Anomaly percentile in [0, 1]; higher = more anomalous."""
        if not self.is_fitted:
            raise RuntimeError("AnomalyDetector must be fitted before predicting.")
        return pd.Series(self._forest.percentile(X), index=X.index, name="iso_score")

    def predict_labels(self, X: pd.DataFrame) -> pd.Series:
        return (self.predict_scores(X) >= 1.0 - self.contamination).rename("iso_flag")

    def explain(self, X: pd.DataFrame, top_n: int = 5) -> list[dict[str, float]]:
        """Top SHAP contributions to the anomaly score for each row of X."""
        if X.empty:
            return []
        try:
            import shap
            explainer = shap.TreeExplainer(self._forest.model)
            values = explainer.shap_values(self._forest.transform(X))
        except Exception as exc:  # shap version / platform issues must never break scoring
            logger.warning("IsolationForest SHAP failed (%s); falling back to robust z-scores.", exc)
            values = self._forest.transform(X)
            return [_top_abs(dict(zip(self.feature_columns, row)), top_n) for row in values]
        # SHAP explains the forest's path length (higher = more normal), so negate.
        return [_top_abs(dict(zip(self.feature_columns, -row)), top_n) for row in values]


class MarketAnomalyDetector:
    """Window-level IsolationForest over market-context features."""

    def __init__(self, config: dict[str, Any]) -> None:
        iso = config.get("isolation_forest", {})
        self._forest = _PercentileForest(
            float(iso.get("market_contamination", 0.05)),
            iso.get("n_estimators", 200),
            "auto",
            iso.get("random_state", 42),
        )
        self.is_fitted = False

    @staticmethod
    def _windows(X: pd.DataFrame) -> pd.DataFrame:
        """One row per market window. Deduplicated on the market VALUES, not on (commodity,
        window_start): cumulative training stacks markets whose keys repeat, and key-based
        dedup kept only the first market's windows."""
        cols = market_columns(X)
        return X.loc[~pd.util.hash_pandas_object(X[cols], index=False).duplicated().to_numpy(), cols]

    def fit(self, X: pd.DataFrame) -> None:
        cols = market_columns(X)
        if not cols:
            return
        win = self._windows(X)
        self._forest.fit(win, cols)
        self.is_fitted = True

    def score(self, X: pd.DataFrame) -> pd.Series:
        """Per-row market-regime anomaly percentile (rows of the same window share it)."""
        if not self.is_fitted:
            return pd.Series(0.0, index=X.index, name="mkt_anomaly_score")
        # Market features are identical for every account in a window: score each distinct row of
        # market values once. Distinct VALUES, not (commodity, window_start) keys: cumulative
        # training stacks many markets whose keys repeat (replay markets all start at tick 0).
        cols = self._forest.columns
        codes, _ = pd.factorize(pd.util.hash_pandas_object(X[cols], index=False))
        rep = pd.Series(np.arange(len(X))).groupby(codes).first().to_numpy()
        per_row = self._forest.percentile(X.iloc[rep])
        return pd.Series(per_row[codes], index=X.index, name="mkt_anomaly_score")


def _top_abs(d: dict[str, float], n: int) -> dict[str, float]:
    items = sorted(d.items(), key=lambda kv: abs(kv[1]), reverse=True)[:n]
    return {k: round(float(v), 4) for k, v in items}

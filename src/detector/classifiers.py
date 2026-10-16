"""
src/detector/classifiers.py
===========================
Supervised layer of the detector (SRS FR-2.3 / FR-2.4).

One independent binary XGBoost classifier per persona (spoofing_vs_rest,
wash_trading_vs_rest, pump_and_dump_vs_rest) trained on trader-level rows:
"was THIS account manipulating in THIS window?". Per-persona models (not one
multiclass model) let an account-window be positive for more than one persona.

Technique choices:
  * All trader behavioural, peer-relative and market-context features are
    offered to every persona model; SHAP shows which ones each model actually
    uses (the old hand-picked per-persona subsets hid useful cross-signals and
    the market-level-only inputs could not localise an account).
  * Class imbalance is handled with scale_pos_weight (negatives/positives),
    not resampling, so SHAP explains the true feature distribution.
  * The decision threshold is tuned per persona on out-of-fold predictions
    (GroupKFold over time blocks, so neighbouring overlapping windows never
    straddle train/validation) to maximise F1 — a fixed 0.5 is badly
    calibrated once scale_pos_weight has re-weighted the classes.
  * SHAP values come from TreeExplainer (exact for tree ensembles).
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold

try:
    import xgboost as xgb
except ImportError:  # pragma: no cover
    xgb = None

logger = logging.getLogger(__name__)

EXCLUDED_COLUMNS = {"window_end", "commodity", "trader_id", "window_start"}


class InsufficientLabelsError(ValueError):
    """Raised when a persona has no positive (or no negative) training rows (SRS FR-2.3)."""


def model_columns(X: pd.DataFrame) -> list[str]:
    return [c for c in X.columns if c not in EXCLUDED_COLUMNS and pd.api.types.is_numeric_dtype(X[c])]


class SupervisedDetector:
    """Per-persona binary XGBoost classifiers with tuned thresholds and SHAP."""

    def __init__(self, config: dict[str, Any]) -> None:
        if xgb is None:
            raise ImportError("xgboost is required for the SupervisedDetector.")
        self.xgb_config = config.get("xgboost", {})
        self.personas = config.get("personas", ["spoofing", "wash_trading", "pump_and_dump"])
        sup = config.get("supervised", {})
        self.min_positives = int(sup.get("min_positives", 5))
        self.require_all_personas = bool(sup.get("require_all_personas", False))
        self.cv_folds = int(sup.get("cv_folds", 3))
        self.block_ticks = int(sup.get("cv_block_ticks", 600))
        self.threshold_bounds = tuple(sup.get("threshold_bounds", [0.25, 0.9]))
        self.threshold_plateau_tol = float(sup.get("threshold_plateau_tol", 0.005))
        self.threshold_keep_tol = float(sup.get("threshold_keep_tol", 0.01))
        self.threshold_max_keep = int(sup.get("threshold_max_keep", 2))
        self._kept_rounds: dict[str, int] = {}   # consecutive retrains each persona's threshold was kept
        self.default_threshold = float(config.get("ensemble", {}).get("supervised_threshold", 0.5))
        self.models: dict[str, Any] = {}
        self.thresholds: dict[str, float] = {}
        self.feature_columns: list[str] = []
        self._explainers: dict[str, Any] = {}

    # Backwards-compatible alias used by older tooling.
    @property
    def _feature_cols(self) -> dict[str, list[str]]:
        return {p: self.feature_columns for p in self.models}

    def _make_model(self, scale_pos_weight: float, n_samples: int) -> "xgb.XGBClassifier":
        c = self.xgb_config
        small = n_samples < 5000
        return xgb.XGBClassifier(
            n_estimators=c.get("n_estimators_small", 150) if small else c.get("n_estimators", 300),
            max_depth=c.get("max_depth_small", 3) if small else c.get("max_depth", 4),
            min_child_weight=c.get("min_child_weight_small", 1) if small else c.get("min_child_weight", 1),
            learning_rate=c.get("learning_rate", 0.05),
            subsample=c.get("subsample", 0.8),
            colsample_bytree=c.get("colsample_bytree", 0.8),
            gamma=c.get("gamma", 0.5),
            reg_alpha=c.get("reg_alpha", 0.1),
            reg_lambda=c.get("reg_lambda", 2.0),
            eval_metric=c.get("eval_metric", "logloss"),
            random_state=c.get("random_state", 42),
            scale_pos_weight=float(np.clip(scale_pos_weight, 1.0, c.get("max_scale_pos_weight", 50.0))),
            n_jobs=-1,
        )

    def _groups(self, X: pd.DataFrame) -> np.ndarray:
        if "window_start" in X.index.names:
            ws = X.index.get_level_values("window_start").to_numpy()
        else:
            ws = np.arange(len(X))
        return ws // max(1, self.block_ticks)

    def _pick_threshold(self, grid: np.ndarray, scores: list[float], previous: float | None,
                        kept: int = 0) -> float:
        """
        Stable choice from the OOF F1 curve. The raw argmax jumped around between
        retrains (spoofing 0.43 -> 0.28 -> 0.48 -> 0.68 -> 0.35) because the curve is a
        flat, noisy plateau. So: smooth over neighbouring grid points, take the centre
        of the near-optimal plateau, and keep the previous round's threshold while it
        is still within `threshold_keep_tol` F1 of the best - but for at most
        `threshold_max_keep` retrains in a row (`kept` so far), so an early threshold
        cannot stay frozen while the data moves on (spoofing sat at 0.75 for 5 rounds).
        """
        s = np.asarray(scores, dtype=float)
        smooth = np.convolve(np.pad(s, 1, mode="edge"), np.ones(3) / 3, mode="valid")
        top = smooth.max()
        plateau = grid[smooth >= top - self.threshold_plateau_tol]
        best = float(grid[np.abs(grid - np.median(plateau)).argmin()])
        if previous is not None and kept < self.threshold_max_keep:
            at_prev = float(np.interp(previous, grid, smooth))
            if at_prev >= top - self.threshold_keep_tol:
                return float(previous)
        return best

    def _tune_threshold(self, Xp: pd.DataFrame, y: np.ndarray, groups: np.ndarray, spw: float,
                        previous: float | None = None, kept: int = 0) -> float:
        pos_groups = np.unique(groups[y == 1])
        n_splits = min(self.cv_folds, len(np.unique(groups)))
        if len(pos_groups) < 2 or n_splits < 2:
            return self.default_threshold
        oof = np.full(len(y), np.nan)
        for tr_idx, va_idx in GroupKFold(n_splits=n_splits).split(Xp, y, groups):
            if y[tr_idx].sum() == 0 or y[tr_idx].sum() == len(tr_idx):
                continue
            m = self._make_model(spw, len(tr_idx))
            m.fit(Xp.iloc[tr_idx], y[tr_idx])
            oof[va_idx] = m.predict_proba(Xp.iloc[va_idx])[:, 1]
        ok = ~np.isnan(oof)
        if ok.sum() == 0 or y[ok].sum() == 0:
            return self.default_threshold
        lo, hi = self.threshold_bounds
        grid = np.linspace(lo, hi, 27)
        scores = [f1_score(y[ok], (oof[ok] >= t).astype(int), zero_division=0) for t in grid]
        best = self._pick_threshold(grid, scores, previous, kept)
        logger.info("    tuned threshold %.2f (OOF F1 %.3f there, max %.3f)", best,
                    float(np.interp(best, grid, scores)), max(scores))
        return best

    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> None:
        """
        Train one classifier per persona.

        Raises:
            InsufficientLabelsError: if require_all_personas is set and a persona
            has fewer than `min_positives` positive rows. Otherwise the persona
            is skipped with an explicit warning (it may simply not have occurred
            in the training rounds yet).
        """
        self.feature_columns = model_columns(X)
        Xm = X[self.feature_columns]
        groups = self._groups(X)
        previous = dict(self.thresholds)   # retraining the same detector: keep thresholds steady
        self.models, self.thresholds, self._explainers = {}, {}, {}
        for persona in self.personas:
            if persona not in y.columns:
                continue
            target = y[persona].astype(int).to_numpy()
            n_pos = int(target.sum())
            n_neg = len(target) - n_pos
            if n_pos < self.min_positives or n_neg == 0:
                msg = (f"[{persona}] only {n_pos} positive / {n_neg} negative training rows "
                       f"(need >= {self.min_positives} positives). Add rounds containing this "
                       f"persona or lower supervised.min_positives.")
                if self.require_all_personas:
                    raise InsufficientLabelsError(msg)
                logger.warning(msg + " Skipping this classifier.")
                continue
            spw = n_neg / n_pos
            logger.info("[%s] training on %d rows (%d positive, scale_pos_weight=%.1f)", persona, len(target), n_pos, spw)
            prev, kept = previous.get(persona), self._kept_rounds.get(persona, 0)
            self.thresholds[persona] = self._tune_threshold(Xm, target, groups, spw, prev, kept)
            self._kept_rounds[persona] = kept + 1 if prev is not None and self.thresholds[persona] == prev else 0
            model = self._make_model(spw, len(target))
            model.fit(Xm, target)
            self.models[persona] = model

    def is_trained(self) -> bool:
        return bool(self.models)

    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=X.index)
        if not self.models:
            return out
        Xm = X.reindex(columns=self.feature_columns, fill_value=0.0)
        for persona, model in self.models.items():
            out[persona] = model.predict_proba(Xm)[:, 1]
        return out

    def predict_flags(self, probs: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            {p: probs[p] >= self.thresholds.get(p, self.default_threshold) for p in probs.columns},
            index=probs.index,
        )

    def shap_values(self, persona: str, X: pd.DataFrame) -> pd.DataFrame:
        """SHAP values (log-odds contributions) of `persona`'s classifier for rows of X."""
        import shap
        if persona not in self._explainers:
            self._explainers[persona] = shap.TreeExplainer(self.models[persona])
        Xm = X.reindex(columns=self.feature_columns, fill_value=0.0)
        vals = self._explainers[persona].shap_values(Xm)
        if isinstance(vals, list):
            vals = vals[-1]
        return pd.DataFrame(vals, index=X.index, columns=self.feature_columns)

    def mean_abs_shap(self, X: pd.DataFrame, max_rows: int = 3000, seed: int = 0) -> dict[str, dict[str, float]]:
        """Global SHAP importance per persona (used for round-to-round SHAP drift, FR-3.4)."""
        if X.empty or not self.models:
            return {}
        sample = X.sample(min(max_rows, len(X)), random_state=seed) if len(X) > max_rows else X
        out = {}
        for persona in self.models:
            sv = self.shap_values(persona, sample).abs().mean()
            out[persona] = {k: round(float(v), 6) for k, v in sv.sort_values(ascending=False).items()}
        return out

"""
src/live/bar_model.py
=====================
Bar-level manipulation model: one binary XGBoost classifier per persona on the
scale-free 1-minute bar features (src/live/bars.py).

Trained each adversarial round on the simulator's training markets of rounds
0..N plus the test markets of rounds 0..N-1 (all labelled, none of them used to
judge round N), and evaluated on round N's held-out test markets, so its stored
metrics say how well anonymous bars alone reveal each persona. Those metrics travel with the model and are printed next to every
real-data prediction: a model that is barely better than chance on simulated
markets, where the answer is known, cannot be trusted more on real ones.

Thresholds are tuned per persona on out-of-fold predictions grouped by market
(no market is ever scored by a model that saw it). With rare positives and weak
signal the F1-best threshold drifts toward "flag everything", so each persona's
threshold is also held to an alert budget: at most ALERT_BUDGET of minutes flagged
out-of-fold, the way a surveillance desk caps alerts to what analysts can review.

Scores are smoothed over SMOOTH_MINUTES (centred, within one commodity's contiguous
bars) before thresholds apply: episodes last 3-15 minutes, so a footprint that
persists across neighbouring minutes is more telling than one noisy minute.
Feature and smoothing experiments: held-out simulator AUC 0.608 -> 0.648 (spoofing
0.64 -> 0.76) over rounds 3-5 with the v2 features, the added training markets and smoothing.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold

from src.live.bars import FEATURES, bar_features, bars_for_market, episode_labels

logger = logging.getLogger(__name__)

PERSONAS = ["spoofing", "wash_trading", "pump_and_dump", "layering"]
MIN_POSITIVES = 10
SMOOTH_MINUTES = 5
ALERT_BUDGET = 0.02              # max share of minutes a persona may flag (out-of-fold)
THRESHOLD_GRID = np.round(np.arange(0.05, 0.96, 0.01), 2)
XGB_PARAMS = dict(n_estimators=250, max_depth=3, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                  min_child_weight=2, reg_lambda=2.0, eval_metric="logloss", random_state=42, n_jobs=4)


def smooth_probs(probs: pd.DataFrame, keys: pd.DataFrame | None, k: int) -> pd.DataFrame:
    """Centred rolling mean of each column within each (commodity, segment) run of bars."""
    if keys is None or k <= 1 or probs.empty:
        return probs
    g = [keys["commodity"].to_numpy(), keys["segment"].to_numpy()]
    return probs.groupby(g).transform(lambda v: v.rolling(k, center=True, min_periods=1).mean())


def market_dataset(market_dirs: list[Path], personas: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, pd.DataFrame]:
    """
    (X, y, groups, keys) over simulated market directories. groups = market index;
    keys = commodity and a segment id unique per market, for smoothing within one market.
    """
    Xs, ys, gs, ks = [], [], [], []
    for i, d in enumerate(market_dirs):
        d = Path(d)
        if not (d / "trade_log.parquet").exists():
            continue
        bars = bars_for_market(d)
        ev_path = d / "manipulation_events.parquet"
        ev = pd.read_parquet(ev_path) if ev_path.exists() else pd.DataFrame()
        Xs.append(bar_features(bars))
        ys.append(episode_labels(bars, ev, personas))
        gs.append(np.full(len(bars), i))
        ks.append(pd.DataFrame({"commodity": bars["commodity"].to_numpy(),
                                "segment": bars["segment"].to_numpy() + 1_000_000 * i}))
    if not Xs:
        return (pd.DataFrame(columns=FEATURES), pd.DataFrame(columns=personas), np.array([]),
                pd.DataFrame(columns=["commodity", "segment"]))
    return (pd.concat(Xs, ignore_index=True), pd.concat(ys, ignore_index=True), np.concatenate(gs),
            pd.concat(ks, ignore_index=True))


def _best_threshold(y: np.ndarray, p: np.ndarray, budget: float = ALERT_BUDGET) -> float:
    """F1-best threshold, raised if needed so no more than `budget` of minutes are flagged."""
    scores = [f1_score(y, p >= t, zero_division=0) for t in THRESHOLD_GRID]
    best = float(THRESHOLD_GRID[int(np.argmax(scores))])
    cap = float(np.quantile(p, 1.0 - budget)) if len(p) else best
    return round(max(best, cap), 4)


def _skill(y: np.ndarray, p: np.ndarray, thr: float) -> dict[str, float | None]:
    if len(np.unique(y)) < 2:
        return {"auc": None, "avg_precision": None, "base_rate": float(y.mean()) if len(y) else None,
                "precision": None, "recall": None, "f1": None, "flag_rate": float((p >= thr).mean()) if len(p) else None}
    flag = p >= thr
    tp = int((flag & (y == 1)).sum())
    return {
        "auc": round(float(roc_auc_score(y, p)), 4),
        "avg_precision": round(float(average_precision_score(y, p)), 4),
        "base_rate": round(float(y.mean()), 4),
        "precision": round(tp / max(int(flag.sum()), 1), 4),
        "recall": round(tp / max(int(y.sum()), 1), 4),
        "f1": round(float(f1_score(y, flag, zero_division=0)), 4),
        "flag_rate": round(float(flag.mean()), 4),
    }


class BarModel:
    """Per-persona classifiers on bar features, with thresholds and simulator skill metrics."""

    def __init__(self, personas: list[str] | None = None) -> None:
        self.personas = list(personas or PERSONAS)
        self.models: dict[str, Any] = {}
        self.thresholds: dict[str, float] = {}
        self.features = list(FEATURES)
        self.metrics: dict[str, Any] = {}
        self.smooth_minutes = SMOOTH_MINUTES

    @property
    def _smooth(self) -> int:
        return int(getattr(self, "smooth_minutes", 1))     # models saved before smoothing existed: none

    # ------------------------------------------------------------------
    def fit(self, X: pd.DataFrame, y: pd.DataFrame, groups: np.ndarray, keys: pd.DataFrame | None = None) -> "BarModel":
        import xgboost as xgb

        X = X[self.features]
        n_groups = len(np.unique(groups))
        oof_metrics = {}
        for p in self.personas:
            yp = y[p].to_numpy() if p in y else np.zeros(len(X), int)
            npos = int(yp.sum())
            if npos < MIN_POSITIVES or npos == len(yp):
                logger.warning("Bar model: %s has %d positive minutes; skipped.", p, npos)
                continue
            spw = min((len(yp) - npos) / max(npos, 1), 50.0)
            params = {**XGB_PARAMS, "scale_pos_weight": spw}
            # Out-of-fold probabilities, grouped by market, choose the threshold.
            oof = np.zeros(len(yp))
            if n_groups >= 2:
                for tr, va in GroupKFold(n_splits=min(3, n_groups)).split(X, yp, groups):
                    if yp[tr].sum() == 0:
                        continue
                    m = xgb.XGBClassifier(**params).fit(X.iloc[tr], yp[tr])
                    oof[va] = m.predict_proba(X.iloc[va])[:, 1]
                oof = smooth_probs(pd.DataFrame({p: oof}), keys, self._smooth)[p].to_numpy()
                self.thresholds[p] = _best_threshold(yp, oof)
                oof_metrics[p] = _skill(yp, oof, self.thresholds[p])
            else:
                self.thresholds[p] = 0.5
            self.models[p] = xgb.XGBClassifier(**params).fit(X, yp)
        self.metrics["train_oof"] = oof_metrics
        self.metrics["n_train_minutes"] = int(len(X))
        self.metrics["n_train_markets"] = int(n_groups)
        return self

    def predict_proba(self, X: pd.DataFrame, keys: pd.DataFrame | None = None) -> pd.DataFrame:
        """Per-persona probabilities; smoothed over neighbouring minutes when `keys` are given."""
        X = X[self.features]
        probs = pd.DataFrame({p: m.predict_proba(X)[:, 1] for p, m in self.models.items()}, index=X.index)
        return smooth_probs(probs, keys, self._smooth)

    def evaluate(self, X: pd.DataFrame, y: pd.DataFrame, keys: pd.DataFrame | None = None) -> dict[str, Any]:
        """Skill on held-out simulated markets; stored as metrics['heldout']."""
        probs = self.predict_proba(X, keys)
        out = {p: _skill(y[p].to_numpy(), probs[p].to_numpy(), self.thresholds[p]) for p in probs.columns if p in y}
        self.metrics["heldout"] = out
        self.metrics["n_heldout_minutes"] = int(len(X))
        return out

    def score(self, bars: pd.DataFrame, X: pd.DataFrame | None = None) -> pd.DataFrame:
        """
        Per-minute scores for a bars table: prob_<persona>, flag, predicted_persona
        (largest margin over its own threshold) and margin.
        """
        X = bar_features(bars) if X is None else X
        keys = bars[["commodity", "segment"]].reset_index(drop=True).set_axis(X.index)
        probs = self.predict_proba(X, keys)
        out = bars[["commodity", "minute", "segment"]].copy()
        if probs.empty:
            out["flag"], out["predicted_persona"], out["margin"] = False, None, 0.0
            return out
        margins = pd.DataFrame({p: probs[p] - self.thresholds[p] for p in probs.columns})
        for p in probs.columns:
            out[f"prob_{p}"] = probs[p].round(4)
        best = margins.to_numpy().argmax(axis=1)
        out["margin"] = margins.to_numpy().max(axis=1).round(4)
        out["flag"] = out["margin"] >= 0
        out["predicted_persona"] = np.where(out["flag"], np.array(list(probs.columns), dtype=object)[best], None)
        return out

    def skill_summary(self) -> dict[str, Any]:
        """Held-out AUC per persona and an overall verdict, for reports."""
        held = self.metrics.get("heldout") or self.metrics.get("train_oof") or {}
        aucs = {p: m.get("auc") for p, m in held.items()}
        vals = [a for a in aucs.values() if a is not None]
        mean_auc = float(np.mean(vals)) if vals else None
        if mean_auc is None:
            level = "unknown"
        elif mean_auc < 0.6:
            level = "near chance"
        elif mean_auc < 0.7:
            level = "weak"
        elif mean_auc < 0.8:
            level = "moderate"
        else:
            level = "good"
        return {"auc": aucs, "mean_auc": None if mean_auc is None else round(mean_auc, 3), "level": level}


def train_round_bar_model(run_dir: Path, round_num: int, personas: list[str] | None = None) -> BarModel:
    """
    Train on the training markets of rounds 0..round_num and the test markets of rounds
    0..round_num-1; evaluate on round_num's test markets (never seen in training).
    """
    run_dir = Path(run_dir)
    personas = list(personas or PERSONAS)
    train_dirs = [run_dir / f"round_{r}" / "train" for r in range(round_num + 1)]
    train_dirs += [d for r in range(round_num) for d in sorted((run_dir / f"round_{r}").glob("test*")) if d.is_dir()]
    rd = run_dir / f"round_{round_num}"
    test_dirs = sorted(d for d in rd.glob("test*") if d.is_dir())
    X, y, g, k = market_dataset(train_dirs, personas)
    model = BarModel(personas).fit(X, y, g, k)
    Xt, yt, _, kt = market_dataset(test_dirs, personas)
    if len(Xt):
        model.evaluate(Xt, yt, kt)
    model.metrics["trained_on_rounds"] = list(range(round_num + 1))
    model.metrics["trained_on_markets"] = [f"{d.parent.name}/{d.name}" for d in train_dirs]
    model.metrics["feature_version"] = 2
    model.metrics["smooth_minutes"] = model.smooth_minutes
    model.metrics["evaluated_on"] = [d.name for d in test_dirs]
    return model

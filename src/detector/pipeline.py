"""
src/detector/pipeline.py
========================
Orchestrates the detector: features → labels → unsupervised + supervised layers
→ ensemble flags with SHAP evidence (SRS FR-2.1 … FR-2.5).

Unit of detection: one (commodity, window, trader_id) row. Ground truth is
trader-level: a row is positive for persona P when the account is one of the
accounts of a P episode and the window overlaps that episode.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.detector.anomaly import AnomalyDetector, MarketAnomalyDetector
from src.detector.classifiers import SupervisedDetector
from src.detector.features import FeatureEngine
from src.utils import io as io_utils

logger = logging.getLogger(__name__)

FEATURES_FILE = "features.parquet"
LABELS_FILE = "labels.parquet"
MIN_OVERLAP_TICKS = 15


def clean_shap_dict(d) -> dict[str, float]:
    """Parquet stores dict columns as structs with the union of all keys; drop the padding."""
    if d is None:
        return {}
    items = d.items() if isinstance(d, dict) else []
    return {k: float(v) for k, v in items if v is not None and not (isinstance(v, float) and np.isnan(v))}


def read_flags(path: Path) -> pd.DataFrame:
    """Read a flags.parquet with top_shap_features / triggered_by restored to plain Python objects."""
    df = pd.read_parquet(path)
    df["top_shap_features"] = df["top_shap_features"].apply(clean_shap_dict)
    # persona_scores exists from flags schema v2.1 on; older files get an empty dict.
    df["persona_scores"] = (df["persona_scores"].apply(clean_shap_dict) if "persona_scores" in df.columns
                            else [{} for _ in range(len(df))])
    df["triggered_by"] = df["triggered_by"].apply(lambda x: list(x) if x is not None else [])
    return df


class DetectorPipeline:
    """End-to-end detector: load → features → labels → fit → score → flags."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.personas = config.get("personas", ["spoofing", "wash_trading", "pump_and_dump"])
        self.feature_engine = FeatureEngine(config)
        self.market_detector = MarketAnomalyDetector(config)
        self.anomaly_detector = AnomalyDetector(config)
        self.supervised_detector = SupervisedDetector(config)
        self.top_n_shap = int(config.get("ensemble", {}).get("top_shap_features", 5))
        self.is_trained = False

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def load_and_preprocess(self, run_dir: Path, *, use_cache: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Build (X, y) for one simulation output directory.

        Features/labels are cached next to the logs (features.parquet,
        labels.parquet) so cumulative multi-round training does not recompute
        every earlier round.
        """
        run_dir = Path(run_dir)
        f_path, l_path = run_dir / FEATURES_FILE, run_dir / LABELS_FILE
        src = run_dir / "order_log.parquet"
        if use_cache and f_path.exists() and l_path.exists() and src.exists() \
                and f_path.stat().st_mtime >= src.stat().st_mtime:
            X = pd.read_parquet(f_path)
            y = pd.read_parquet(l_path)
            if set(self.personas).issubset(y.columns):
                return X, y[self.personas]

        logger.info("Computing features for %s", run_dir)
        trade_df = pd.read_parquet(run_dir / "trade_log.parquet")
        order_df = pd.read_parquet(run_dir / "order_log.parquet")
        manip_df = pd.read_parquet(run_dir / "manipulation_events.parquet")
        X = self.feature_engine.generate_features(trade_df, order_df)
        y = self.generate_labels(X, manip_df)
        if not X.empty:
            X.to_parquet(f_path)
            y.to_parquet(l_path)
        return X, y

    def generate_labels(self, X: pd.DataFrame, manip_df: pd.DataFrame) -> pd.DataFrame:
        """Trader-level ground truth: account ∈ episode accounts AND window overlaps episode."""
        y = pd.DataFrame(0, index=X.index, columns=self.personas, dtype=int)
        if X.empty or manip_df.empty:
            return y
        rows = X.reset_index()[["commodity", "window_start", "trader_id", "window_end"]]
        rows["_row"] = np.arange(len(rows))
        ev = manip_df[["commodity", "persona", "start_tick", "end_tick", "trader_ids"]].copy()
        ev["trader_ids"] = ev["trader_ids"].apply(list)
        ev = ev.explode("trader_ids").rename(columns={"trader_ids": "trader_id"})
        j = rows.merge(ev, on=["commodity", "trader_id"], how="inner")
        overlap = np.minimum(j["window_end"], j["end_tick"]) - np.maximum(j["window_start"], j["start_tick"])
        j = j[overlap >= MIN_OVERLAP_TICKS]
        values = y.to_numpy()
        for persona in self.personas:
            hit = j.loc[j["persona"] == persona, "_row"].unique()
            values[hit, self.personas.index(persona)] = 1
        return pd.DataFrame(values, index=X.index, columns=self.personas)

    # Backwards-compatible name
    _generate_labels = generate_labels

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def _with_context(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X["mkt_anomaly_score"] = self.market_detector.score(X).to_numpy()
        return X

    def train(self, X: pd.DataFrame, y: pd.DataFrame, save_dir: Path | None = None) -> None:
        """Fit the market IF, the trader IF and the per-persona classifiers."""
        self.market_detector.fit(X)
        Xc = self._with_context(X)
        self.anomaly_detector.fit(Xc)
        self.supervised_detector.fit(Xc, y)
        self.is_trained = True
        if save_dir is not None:
            self.save_importances(Xc, Path(save_dir))

    def save_importances(self, X: pd.DataFrame, save_dir: Path) -> None:
        """Persist per-persona SHAP importance (FR-3.4) and native gain importance."""
        save_dir.mkdir(parents=True, exist_ok=True)
        shap_imp = self.supervised_detector.mean_abs_shap(X)
        with open(save_dir / "shap_importance.json", "w") as f:
            json.dump(shap_imp, f, indent=2)
        gain = {
            p: {c: float(v) for c, v in zip(self.supervised_detector.feature_columns, m.feature_importances_)}
            for p, m in self.supervised_detector.models.items()
        }
        with open(save_dir / "feature_importances.json", "w") as f:
            json.dump(gain, f, indent=2)
        with open(save_dir / "thresholds.json", "w") as f:
            json.dump(self.supervised_detector.thresholds, f, indent=2)

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def score(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Ensemble scoring (FR-2.5). Columns:
          iso_score / iso_flag            trader-level IsolationForest percentile / flag
          mkt_anomaly_score               market-regime IsolationForest percentile
          prob_<p> / flag_<p>             per-persona probability / tuned-threshold flag
          sup_flag, final_flag, final_score, predicted_persona, triggered_by
        """
        if not self.is_trained:
            raise RuntimeError("DetectorPipeline.score() called before train().")
        Xc = self._with_context(X)
        res = pd.DataFrame(index=X.index)
        res["window_end"] = X["window_end"].to_numpy() if "window_end" in X else 0
        res["mkt_anomaly_score"] = Xc["mkt_anomaly_score"]
        res["iso_score"] = self.anomaly_detector.predict_scores(Xc)
        res["iso_flag"] = self.anomaly_detector.predict_labels(Xc)

        probs = self.supervised_detector.predict_proba(Xc)
        flags = self.supervised_detector.predict_flags(probs) if not probs.empty else pd.DataFrame(index=X.index)
        for p in self.personas:
            res[f"prob_{p}"] = probs[p] if p in probs else 0.0
            res[f"flag_{p}"] = flags[p] if p in flags else False

        flag_cols = [f"flag_{p}" for p in self.personas]
        prob_cols = [f"prob_{p}" for p in self.personas]
        res["sup_flag"] = res[flag_cols].any(axis=1)
        # Margin over each persona's own threshold picks the most confident persona.
        margins = np.column_stack([
            res[f"prob_{p}"] - self.supervised_detector.thresholds.get(p, 1.0) for p in self.personas
        ])
        best = np.argmax(margins, axis=1)
        res["predicted_persona"] = np.where(
            res["sup_flag"], np.array(self.personas, dtype=object)[best],
            np.where(res["iso_flag"], "anomaly", None),
        )
        rule = self.config.get("ensemble", {}).get("combination_rule", "or")
        if rule == "weighted":
            e = self.config.get("ensemble", {})
            res["final_score"] = (e.get("iso_weight", 0.4) * res["iso_score"]
                                  + e.get("supervised_weight", 0.6) * res[prob_cols].max(axis=1)).clip(0, 1)
            res["final_flag"] = res["sup_flag"] | (res["final_score"] > 0.5)
        else:
            res["final_score"] = np.maximum(res[prob_cols].max(axis=1), res["iso_score"] * res["iso_flag"]).clip(0, 1)
            res["final_flag"] = res["sup_flag"] | res["iso_flag"]
        res["triggered_by"] = [
            ([f"{p}_classifier" for p in self.personas if row[f"flag_{p}"]]
             + (["isolation_forest"] if row["iso_flag"] else []))
            for _, row in res[flag_cols + ["iso_flag"]].iterrows()
        ]
        return res

    def build_flags(self, X: pd.DataFrame, results: pd.DataFrame) -> pd.DataFrame:
        """
        flags table (FlagsSchema): one row per (commodity, window, trader) with the
        top SHAP attributions of the classifier that drove the flag — or of the
        IsolationForest when only the unsupervised layer fired.
        """
        Xc = self._with_context(X)
        top: list[dict] = [{} for _ in range(len(results))]
        pos = {key: i for i, key in enumerate(results.index)}
        flagged = results[results["final_flag"]]
        for persona in self.personas:
            idx = flagged.index[flagged["predicted_persona"] == persona]
            if len(idx) and persona in self.supervised_detector.models:
                sv = self.supervised_detector.shap_values(persona, Xc.loc[idx])
                for key, row in zip(idx, sv.to_numpy()):
                    order = np.argsort(-np.abs(row))[: self.top_n_shap]
                    top[pos[key]] = {sv.columns[j]: round(float(row[j]), 4) for j in order}
        idx = flagged.index[flagged["predicted_persona"] == "anomaly"]
        if len(idx):
            for key, d in zip(idx, self.anomaly_detector.explain(Xc.loc[idx], self.top_n_shap)):
                top[pos[key]] = d

        out = results.reset_index()[["commodity", "window_start", "trader_id"]].copy()
        out["window_start"] = out["window_start"].astype("int64")
        out["window_end"] = results["window_end"].to_numpy().astype("int64")
        out["final_flag"] = results["final_flag"].to_numpy().astype(bool)
        out["final_score"] = results["final_score"].to_numpy().astype(float)
        out["iso_score"] = results["iso_score"].to_numpy().astype(float)
        out["predicted_persona"] = results["predicted_persona"].to_numpy()
        out["triggered_by"] = list(results["triggered_by"])
        out["top_shap_features"] = list(top)
        trained = [p for p in self.personas if p in self.supervised_detector.models]
        probs = results[[f"prob_{p}" for p in trained]].to_numpy() if trained else None
        out["persona_scores"] = [
            {p: round(float(v), 4) for p, v in zip(trained, row)} for row in probs
        ] if probs is not None else [{} for _ in range(len(out))]
        return out

    def write_flags(self, flags: pd.DataFrame, path: Path) -> Path:
        return io_utils.write_parquet(flags, path, "flags", overwrite=True)

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
# Activity-based labels: order-id audit tags (simulator ground truth, never seen by the detector)
# that mark an account's manipulative orders, per persona. A window is positive only if the account
# placed at least one of them inside it. Quiet accumulation ("a"), pump-and-dump decoys ("d") and wash
# camouflage ("c") are left out on purpose: they are built to be indistinguishable from ordinary
# trading, and labelling them taught the classifiers to flag ordinary traders. Replayed real cases use
# "s"/"g"/"f" (spoofing and layering scripts), "w" (wash crosses) and "p" (trade programmes).
ACTIVITY_TAGS = {
    "spoofing": {"s", "g", "f"},
    "layering": {"l", "g", "s"},
    "wash_trading": {"w"},
    "pump_and_dump": {"i", "x", "u", "p"},
}


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
        from src.detector.features import FEATURE_VERSION
        v_path = run_dir / "features.version"
        cached_version = v_path.read_text().strip() if v_path.exists() else "1"
        version = f"{FEATURE_VERSION}:{self.label_mode}"
        if use_cache and f_path.exists() and l_path.exists() and src.exists() \
                and f_path.stat().st_mtime >= src.stat().st_mtime and cached_version == version:
            X = pd.read_parquet(f_path)
            y = pd.read_parquet(l_path)
            from src.detector.features import SELF_Z_COLUMNS
            current = all(f"self_z_{c}" in X.columns for c in SELF_Z_COLUMNS)   # cache from an older feature set?
            if current and set(self.personas).issubset(y.columns):
                return X, y[self.personas]

        logger.info("Computing features for %s", run_dir)
        trade_df = pd.read_parquet(run_dir / "trade_log.parquet")
        order_df = pd.read_parquet(run_dir / "order_log.parquet")
        manip_df = pd.read_parquet(run_dir / "manipulation_events.parquet")
        X = self.feature_engine.generate_features(trade_df, order_df)
        y = self.generate_labels(X, manip_df, order_df)
        if not X.empty:
            X.to_parquet(f_path)
            y.to_parquet(l_path)
            v_path.write_text(version)
        return X, y

    @property
    def label_mode(self) -> str:
        """'activity' (default): windows where the account placed manipulative orders; 'span': the whole episode."""
        return str(self.config.get("labels", {}).get("mode", "activity"))

    def generate_labels(self, X: pd.DataFrame, manip_df: pd.DataFrame,
                        order_df: pd.DataFrame | None = None) -> pd.DataFrame:
        """
        Trader-level ground truth.

        activity (default, needs order_df): the account is one of the episode's accounts AND it
            placed at least one of the persona's manipulative orders (ACTIVITY_TAGS) inside the
            window, within the episode.
        span (or no order log given): the account is one of the episode's accounts AND the
            window overlaps the episode by MIN_OVERLAP_TICKS.
        """
        y = pd.DataFrame(0, index=X.index, columns=self.personas, dtype=int)
        if X.empty or manip_df.empty:
            return y
        if self.label_mode == "activity" and order_df is not None and not order_df.empty:
            return self._activity_labels(X, manip_df, order_df, y)
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

    def _activity_labels(self, X: pd.DataFrame, manip_df: pd.DataFrame, order_df: pd.DataFrame,
                         y: pd.DataFrame) -> pd.DataFrame:
        placed = order_df[order_df["event_type"] == "placed"][["order_id", "trader_id", "tick", "commodity"]]
        ev = manip_df[["commodity", "persona", "start_tick", "end_tick", "trader_ids"]].copy()
        ev["trader_ids"] = ev["trader_ids"].apply(list)
        ev = ev.explode("trader_ids").rename(columns={"trader_ids": "trader_id"})
        ev = ev[ev["persona"].isin(self.personas)]
        acts = placed[placed["trader_id"].isin(set(ev["trader_id"]))].copy()
        if acts.empty:
            return y
        acts["tag"] = acts["order_id"].astype(str).str.extract(r"-\d+([a-z]*)$")[0].fillna("")
        acts = acts.merge(ev, on=["commodity", "trader_id"], how="inner")
        acts = acts[acts["tick"].between(acts["start_tick"], acts["end_tick"])]
        keep = np.array([t in ACTIVITY_TAGS.get(p, set()) for t, p in zip(acts["tag"], acts["persona"])], dtype=bool)
        acts = acts[keep]
        if acts.empty:
            return y
        rows = X.reset_index()[["commodity", "window_start", "trader_id", "window_end"]]
        rows["_row"] = np.arange(len(rows))
        rows = rows[rows["trader_id"].isin(set(acts["trader_id"]))]
        values = y.to_numpy()
        for (com, tid, persona), g in acts.groupby(["commodity", "trader_id", "persona"]):
            ticks = np.sort(g["tick"].to_numpy())
            w = rows[(rows["commodity"] == com) & (rows["trader_id"] == tid)]
            if w.empty:
                continue
            lo = np.searchsorted(ticks, w["window_start"].to_numpy(), side="left")
            hi = np.searchsorted(ticks, w["window_end"].to_numpy(), side="right")
            hit = w["_row"].to_numpy()[hi > lo]
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

    def train(self, X: pd.DataFrame, y: pd.DataFrame, save_dir: Path | None = None,
              sample_weight: np.ndarray | None = None) -> None:
        """Fit the market IF, the trader IF and the per-persona classifiers (optionally row-weighted)."""
        self.market_detector.fit(X)
        Xc = self._with_context(X)
        self.anomaly_detector.fit(Xc)
        self.supervised_detector.fit(Xc, y, sample_weight=sample_weight)
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
        e = self.config.get("ensemble", {})
        rule = e.get("combination_rule", "or")
        # An IsolationForest flag on its own is often a legitimate but unusual firm (HFT,
        # execution algorithm). With "or_corroborated" it becomes an alert only if a
        # classifier is at least `min_prob_ratio` of the way to its threshold, or the
        # account deviates strongly from its own history; otherwise it is a review item.
        corr = e.get("iso_corroboration", {})
        ratio = np.column_stack([
            res[f"prob_{p}"] / max(self.supervised_detector.thresholds.get(p, 1.0), 1e-6) for p in self.personas
        ]).max(axis=1)
        self_cols = [c for c in Xc.columns if c.startswith("self_z_")]
        # Self-history corroboration needs `min_self_z_cols` features deviating at once: one
        # feature (usually order size) jumps whenever an execution algo starts a new parent
        # order, and that single-feature route made nearly all of their false alerts.
        n_dev = ((Xc[self_cols].abs() >= corr.get("min_self_z", 6.0)).sum(axis=1).to_numpy()
                 if self_cols else np.zeros(len(Xc)))
        corroborated = (ratio >= corr.get("min_prob_ratio", 0.5)) | (n_dev >= corr.get("min_self_z_cols", 1))
        if rule == "weighted":
            e = self.config.get("ensemble", {})
            res["final_score"] = (e.get("iso_weight", 0.4) * res["iso_score"]
                                  + e.get("supervised_weight", 0.6) * res[prob_cols].max(axis=1)).clip(0, 1)
            res["final_flag"] = res["sup_flag"] | (res["final_score"] > 0.5)
        else:
            iso_alert = res["iso_flag"] & (corroborated if rule == "or_corroborated" else True)
            res["final_score"] = np.maximum(res[prob_cols].max(axis=1), res["iso_score"] * iso_alert).clip(0, 1)
            res["final_flag"] = res["sup_flag"] | iso_alert
        res["alert_tier"] = np.where(res["final_flag"], "alert", np.where(res["iso_flag"], "review", "none"))
        res.loc[res["alert_tier"] == "review", "predicted_persona"] = None
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
        out["alert_tier"] = results["alert_tier"].to_numpy() if "alert_tier" in results else             np.where(results["final_flag"], "alert", "none")
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

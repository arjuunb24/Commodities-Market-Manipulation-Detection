"""
src/detector/pipeline.py
========================
Orchestrates the Detector Subsystem ML workflow.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from src.detector.features import FeatureEngine
from src.detector.anomaly import AnomalyDetector
from src.detector.classifiers import SupervisedDetector
from src.utils import io as io_utils

logger = logging.getLogger(__name__)

class DetectorPipeline:
    """
    End-to-end pipeline for the Detector Subsystem.
    Loads raw data -> Extracts Features -> Generates Labels -> Fits/Scores Models.
    """
    
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.feature_engine = FeatureEngine(config)
        self.anomaly_detector = AnomalyDetector(config)
        self.supervised_detector = SupervisedDetector(config)
        
    def load_and_preprocess(self, run_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Loads the Parquet files for a given run and processes them into X and y.
        """
        logger.info(f"Loading run data from {run_dir}")
        trade_df = pd.read_parquet(run_dir / "trade_log.parquet")
        order_df = pd.read_parquet(run_dir / "order_log.parquet")
        manip_df = pd.read_parquet(run_dir / "manipulation_events.parquet")
        
        # 1. Generate Feature Matrix X
        X = self.feature_engine.generate_features(trade_df, order_df)
        
        # 2. Generate Labels y
        y = self._generate_labels(X, manip_df)
        
        return X, y
        
    def _generate_labels(self, X: pd.DataFrame, manip_df: pd.DataFrame) -> pd.DataFrame:
        """
        Cross-references rolling window timestamps with the raw tick-based
        manipulation events to generate binary labels for each persona.
        """
        personas = self.config.get("personas", ["spoofing", "wash_trading", "pump_and_dump"])
        
        # Initialize labels DataFrame with 0
        y = pd.DataFrame(0, index=X.index, columns=personas)
        
        if manip_df.empty:
            return y
            
        # We need to map the start_tick/end_tick of manip_df to timestamps to overlap with X
        # X index is (commodity, timestamp).
        # We assume base_time is 2026-01-01 00:00:00 as defined in FeatureEngine.
        base_time = pd.Timestamp("2026-01-01 00:00:00")
        
        manip_df = manip_df.copy()
        manip_df["start_time"] = base_time + pd.to_timedelta(manip_df["start_tick"], unit="s")
        manip_df["end_time"] = base_time + pd.to_timedelta(manip_df["end_tick"], unit="s")
        
        # For each rolling window in X, it covers [timestamp - window, timestamp].
        # If any manipulation event overlaps this window, it's a positive label.
        window_timedelta = pd.to_timedelta(self.feature_engine.window)
        
        # To avoid massive cross-joins, we iterate through X index (which is fast enough for typical ABM sizes)
        # For large data, merge_asof or interval indexing is better, but iteration works for our scale.
        
        # Optimize by grouping manipulation events by commodity
        manip_by_comm = dict(tuple(manip_df.groupby("commodity")))
        
        y_dict = y.to_dict("index")
        
        for (comm, ts) in X.index:
            if comm not in manip_by_comm:
                continue
                
            window_start = ts - window_timedelta
            window_end = ts
            
            comm_events = manip_by_comm[comm]
            # Overlap condition: event starts before window ends AND event ends after window starts
            overlap = comm_events[(comm_events["start_time"] <= window_end) & (comm_events["end_time"] >= window_start)]
            
            if not overlap.empty:
                for persona in overlap["persona"].unique():
                    # Clean persona name (e.g. spoofer_0 -> spoofing)
                    clean_persona = "spoofing" if "spoof" in persona else \
                                    "wash_trading" if "wash" in persona else \
                                    "pump_and_dump" if "pump" in persona else \
                                    "layering" if "layer" in persona else persona
                                    
                    if clean_persona in y_dict[(comm, ts)]:
                        y_dict[(comm, ts)][clean_persona] = 1
                        
        return pd.DataFrame.from_dict(y_dict, orient="index")
        
    def train(self, X: pd.DataFrame, y: pd.DataFrame) -> None:
        """Trains both the unsupervised and supervised layers."""
        self.anomaly_detector.fit(X)
        self.supervised_detector.fit(X, y)
        
    def score(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Runs the ensemble scorer using both layers.
        Returns a DataFrame with isolation_score, supervised probabilities, and final flags.
        Falls back to ISO-only results if no supervised models were trained.
        """
        results = pd.DataFrame(index=X.index)

        # 1. Unsupervised
        results["iso_score"] = self.anomaly_detector.predict_scores(X)
        results["iso_flag"] = self.anomaly_detector.predict_labels(X)

        # 2. Supervised (graceful fallback if no models trained)
        sup_probs = pd.DataFrame(index=X.index)
        if self.supervised_detector.is_trained():
            sup_probs = self.supervised_detector.predict_proba(X)
            for col in sup_probs.columns:
                results[f"prob_{col}"] = sup_probs[col]
        else:
            logger.warning("No supervised models trained. Using ISO-only ensemble.")

        # Overall supervised probability is the max across personas
        results["sup_max_prob"] = sup_probs.max(axis=1) if not sup_probs.empty else 0.0

        # 3. Ensemble Rule
        ensemble_config = self.config.get("ensemble", {})
        rule = ensemble_config.get("combination_rule", "or")
        sup_thresh = ensemble_config.get("supervised_threshold", 0.5)

        results["sup_flag"] = results["sup_max_prob"] > sup_thresh

        if rule == "or":
            results["ensemble_flag"] = results["iso_flag"] | results["sup_flag"]
        elif rule == "weighted":
            w_iso = ensemble_config.get("iso_weight", 0.4)
            w_sup = ensemble_config.get("supervised_weight", 0.6)
            norm_iso = (results["iso_score"] - results["iso_score"].min()) / (results["iso_score"].max() - results["iso_score"].min() + 1e-9)
            results["ensemble_score"] = (norm_iso * w_iso) + (results["sup_max_prob"] * w_sup)
            results["ensemble_flag"] = results["ensemble_score"] > 0.5

        return results

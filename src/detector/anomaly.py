"""
src/detector/anomaly.py
=======================
Unsupervised Anomaly Detection layer.

Provides a general-purpose IsolationForest plus persona-specific detectors:
 - Wash Trading: LocalOutlierFactor (density-based, catches colluding clusters)
 - Pump & Dump: EllipticEnvelope (tail-event Gaussian, catches extreme price/volume spikes)
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope

logger = logging.getLogger(__name__)

# Features that are most informative for each persona's unsupervised detector
WASH_TRADING_FEATURES = [
    "stateful_circular_volume_ratio", "net_position_ratio",
    "unique_counterparty_ratio", "trade_size_cv", "volume_concentration"
]

PUMP_DUMP_FEATURES = [
    "volume_spike_ratio", "abnormal_return_z",
    "price_momentum", "price_range", "price_volatility", "topk_buy_imbalance"
]


class AnomalyDetector:
    """
    Multi-layer unsupervised detection:
    - General IsolationForest for all personas (spoofing, layering, etc.)
    - LocalOutlierFactor for wash_trading (density-based neighborhood analysis)
    - EllipticEnvelope for pump_and_dump (Gaussian tail-event detection)
    """
    
    def __init__(self, config: dict[str, Any]) -> None:
        """
        Args:
            config: Detector config mapping (loaded from detector_config.yaml)
        """
        iso_config = config.get("isolation_forest", {})
        
        # General-purpose anomaly detector
        self.model = IsolationForest(
            contamination=iso_config.get("contamination", 0.05),
            n_estimators=iso_config.get("n_estimators", 200),
            max_samples=iso_config.get("max_samples", "auto"),
            random_state=iso_config.get("random_state", 42),
            n_jobs=-1
        )
        
        # Persona-specific: Wash Trading
        # LOF detects "local" clusters of highly similar trades (colluding pairs).
        # n_neighbors=10 provides stable neighborhood estimates for small datasets.
        self._lof = LocalOutlierFactor(
            n_neighbors=10,
            contamination=iso_config.get("contamination", 0.05),
            novelty=True,  # novelty=True allows predict() on new data
            n_jobs=-1
        )
        
        # Persona-specific: Pump & Dump
        # EllipticEnvelope fits a robust Gaussian to "normal" data.
        # High reconstruction error → extreme tail event (pump/dump).
        self._ee = EllipticEnvelope(
            contamination=iso_config.get("contamination", 0.05),
            random_state=iso_config.get("random_state", 42),
            support_fraction=0.9
        )

        self.is_fitted = False
        self._lof_features: list[str] = []
        self._ee_features: list[str] = []
        
    def fit(self, X: pd.DataFrame) -> None:
        """
        Fits all three unsupervised models on the feature matrix.
        """
        logger.info(f"Fitting IsolationForest on {len(X)} samples...")
        self.model.fit(X)
        
        # Fit persona-specific detectors on their feature subsets
        self._lof_features = [f for f in WASH_TRADING_FEATURES if f in X.columns]
        if self._lof_features:
            logger.info(f"Fitting LocalOutlierFactor (wash_trading) on features: {self._lof_features}")
            self._lof.fit(X[self._lof_features])
            
        self._ee_features = [f for f in PUMP_DUMP_FEATURES if f in X.columns]
        if self._ee_features and len(X) > len(self._ee_features):
            logger.info(f"Fitting EllipticEnvelope (pump_and_dump) on features: {self._ee_features}")
            try:
                self._ee.fit(X[self._ee_features])
            except Exception as e:
                logger.warning(f"EllipticEnvelope fitting failed (likely singular covariance): {e}. Skipping.")
                self._ee_features = []

        self.is_fitted = True
        
    def predict_scores(self, X: pd.DataFrame) -> pd.Series:
        """
        Returns continuous anomaly scores from the general IsolationForest.
        Higher scores = more anomalous.
        """
        if not self.is_fitted:
            raise RuntimeError("AnomalyDetector must be fitted before predicting.")
            
        raw_scores = self.model.score_samples(X)
        inverted = -raw_scores
        return pd.Series(inverted, index=X.index, name="iso_score")

    def predict_labels(self, X: pd.DataFrame) -> pd.Series:
        """
        Returns boolean flags from the general IsolationForest.
        True = anomalous.
        """
        if not self.is_fitted:
            raise RuntimeError("AnomalyDetector must be fitted before predicting.")
            
        preds = self.model.predict(X)
        return pd.Series(preds == -1, index=X.index, name="is_anomaly")

    def predict_wash_trading_scores(self, X: pd.DataFrame) -> pd.Series:
        """
        Returns LOF anomaly scores specifically tuned for wash trading features.
        Higher = more likely wash trading.
        """
        if not self._lof_features or not self.is_fitted:
            return pd.Series(0.0, index=X.index, name="lof_score")
        
        try:
            available = [f for f in self._lof_features if f in X.columns]
            if not available:
                return pd.Series(0.0, index=X.index, name="lof_score")
            raw = self._lof.score_samples(X[available])
            return pd.Series(-raw, index=X.index, name="lof_score")  # Higher = more anomalous
        except Exception as e:
            logger.warning(f"LOF prediction failed: {e}")
            return pd.Series(0.0, index=X.index, name="lof_score")

    def predict_pump_dump_scores(self, X: pd.DataFrame) -> pd.Series:
        """
        Returns EllipticEnvelope anomaly scores specifically tuned for pump & dump features.
        Higher = more likely pump & dump.
        """
        if not self._ee_features or not self.is_fitted:
            return pd.Series(0.0, index=X.index, name="ee_score")
        
        try:
            available = [f for f in self._ee_features if f in X.columns]
            if not available:
                return pd.Series(0.0, index=X.index, name="ee_score")
            raw = self._ee.score_samples(X[available])
            return pd.Series(-raw, index=X.index, name="ee_score")  # Higher = more anomalous
        except Exception as e:
            logger.warning(f"EllipticEnvelope prediction failed: {e}")
            return pd.Series(0.0, index=X.index, name="ee_score")


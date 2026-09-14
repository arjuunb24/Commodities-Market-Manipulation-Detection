"""
src/detector/anomaly.py
=======================
Unsupervised Anomaly Detection using Isolation Forest.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd
from sklearn.ensemble import IsolationForest

logger = logging.getLogger(__name__)

class AnomalyDetector:
    """
    Unsupervised detection layer designed to flag windows with extreme
    deviations in OTR, Cancel Rate, or Volume Concentration.
    """
    
    def __init__(self, config: dict[str, Any]) -> None:
        """
        Args:
            config: Detector config mapping (loaded from detector_config.yaml)
        """
        iso_config = config.get("isolation_forest", {})
        
        self.model = IsolationForest(
            contamination=iso_config.get("contamination", 0.05),
            n_estimators=iso_config.get("n_estimators", 200),
            max_samples=iso_config.get("max_samples", "auto"),
            random_state=iso_config.get("random_state", 42),
            n_jobs=-1
        )
        self.is_fitted = False
        
    def fit(self, X: pd.DataFrame) -> None:
        """
        Fits the Isolation Forest on the feature matrix.
        Typically, this should be fitted on "normal" (non-manipulated) data,
        but since it's an anomaly detector, it can also be fitted on contaminated data.
        """
        logger.info(f"Fitting IsolationForest on {len(X)} samples...")
        self.model.fit(X)
        self.is_fitted = True
        
    def predict_scores(self, X: pd.DataFrame) -> pd.Series:
        """
        Returns continuous anomaly scores.
        Scikit-learn IsolationForest returns negative scores for anomalies.
        We invert this so that higher scores = more anomalous (0 to 1 scaling).
        """
        if not self.is_fitted:
            raise RuntimeError("AnomalyDetector must be fitted before predicting.")
            
        # score_samples returns the opposite of the anomaly score defined in the original paper.
        # Lower values = more abnormal.
        raw_scores = self.model.score_samples(X)
        
        # Invert so higher = more anomalous. (Typical range is [-0.5, 0.5] approx).
        # We can map it loosely to [0, 1] for the ensemble.
        inverted = -raw_scores
        
        # Normalize to 0-1 range based on empirical min/max or simple scaling.
        # For a robust approach, we just return the raw inverted score, 
        # and pipeline will handle thresholds.
        return pd.Series(inverted, index=X.index, name="iso_score")

    def predict_labels(self, X: pd.DataFrame) -> pd.Series:
        """
        Returns boolean flags: True if anomalous, False otherwise.
        """
        if not self.is_fitted:
            raise RuntimeError("AnomalyDetector must be fitted before predicting.")
            
        # predict returns 1 for inliers, -1 for outliers
        preds = self.model.predict(X)
        return pd.Series(preds == -1, index=X.index, name="is_anomaly")

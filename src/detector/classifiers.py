"""
src/detector/classifiers.py
===========================
Supervised Classification layer using XGBoost.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd
import numpy as np

# Lazy import so we don't crash if xgboost isn't installed during simple test runs
try:
    import xgboost as xgb
except ImportError:
    xgb = None

logger = logging.getLogger(__name__)

class SupervisedDetector:
    """
    Trains and orchestrates binary XGBoost classifiers for each 
    manipulation persona defined in the config.
    """
    
    def __init__(self, config: dict[str, Any]) -> None:
        """
        Args:
            config: Detector config mapping (loaded from detector_config.yaml)
        """
        if xgb is None:
            raise ImportError("xgboost is required for the SupervisedDetector.")
            
        self.xgb_config = config.get("xgboost", {})
        self.personas = config.get("personas", ["spoofing", "wash_trading", "pump_and_dump"])
        
        # We will hold one XGBClassifier per persona
        self.models: dict[str, xgb.XGBClassifier] = {}
        
    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> None:
        """
        Trains binary classifiers for each persona.
        
        Args:
            X: Feature matrix from FeatureEngine.
            y: Boolean DataFrame with columns matching self.personas. 
               Index must match X.
        """
        logger.info(f"Training supervised models on {len(X)} samples for personas: {self.personas}")
        
        for persona in self.personas:
            if persona not in y.columns:
                logger.warning(f"Persona '{persona}' not found in labels DataFrame. Skipping.")
                continue
                
            y_target = y[persona].astype(int)
            
            # Dynamic scale_pos_weight for severe class imbalance
            n_pos = y_target.sum()
            n_neg = len(y_target) - n_pos
            
            if n_pos == 0 or n_neg == 0:
                logger.warning(f"[{persona}] Cannot train XGBoost: found {n_pos} positive and {n_neg} negative examples. Skipping.")
                continue
                
            scale_pos_weight = (n_neg / n_pos) if n_pos > 0 else 1.0
            
            logger.info(f"[{persona}] Class distribution: {n_pos} positive, {n_neg} negative (scale_pos_weight={scale_pos_weight:.2f})")
            
            model = xgb.XGBClassifier(
                n_estimators=self.xgb_config.get("n_estimators", 300),
                max_depth=self.xgb_config.get("max_depth", 5),
                learning_rate=self.xgb_config.get("learning_rate", 0.05),
                subsample=self.xgb_config.get("subsample", 0.8),
                colsample_bytree=self.xgb_config.get("colsample_bytree", 0.8),
                eval_metric=self.xgb_config.get("eval_metric", "logloss"),
                random_state=self.xgb_config.get("random_state", 42),
                scale_pos_weight=scale_pos_weight,
                n_jobs=-1
            )
            
            model.fit(X, y_target)
            self.models[persona] = model
            logger.info(f"[{persona}] Training complete.")
            
    def is_trained(self) -> bool:
        """Returns True if at least one supervised model was successfully trained."""
        return bool(self.models)

    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Returns probability of manipulation for each persona.

        Returns:
            DataFrame with columns matching trained personas.
            Returns an empty DataFrame if no models were trained.
        """
        if not self.models:
            return pd.DataFrame(index=X.index)

        results = pd.DataFrame(index=X.index)

        for persona, model in self.models.items():
            # predict_proba returns [P(class=0), P(class=1)]
            probs = model.predict_proba(X)
            results[persona] = probs[:, 1]

        return results

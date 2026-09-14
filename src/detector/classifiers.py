"""
src/detector/classifiers.py
===========================
Supervised Classification layer.

Uses persona-specific models:
- spoofing: XGBoost (tree-based, handles high cancel_rate spikes)
- wash_trading: LightGBM (leaf-wise tree growth, faster at behavioral uniformity features)
- pump_and_dump: XGBoost + StandardScaler (z-normalized price features)
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Lazy imports so we don't crash if packages aren't installed during simple test runs
try:
    import xgboost as xgb
except ImportError:
    xgb = None

logger = logging.getLogger(__name__)

# Features that benefit from z-score normalization for pump & dump
PUMP_DUMP_SCALE_FEATURES = [
    "volume_spike_ratio", "abnormal_return_z", "price_momentum",
    "price_range", "price_volatility", "price_return", "topk_buy_imbalance"
]


class SupervisedDetector:
    """
    Trains and orchestrates binary classifiers for each 
    manipulation persona defined in the config.
    
    Persona-specific models:
    - spoofing/layering/wash_trading: XGBoost
    - pump_and_dump: XGBoost with StandardScaler on price features
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
        
        # One model per persona
        self.models: dict[str, Any] = {}
        # Scaler for pump_and_dump only
        self._pd_scaler: StandardScaler | None = None
        
    def _make_xgb(self, scale_pos_weight: float, n_samples: int) -> "xgb.XGBClassifier":
        """Builds an XGBClassifier with dynamic hyperparameters."""
        if n_samples < 5000:
            n_est = self.xgb_config.get("n_estimators_small", 50)
            max_d = self.xgb_config.get("max_depth_small", 2)
            min_c = self.xgb_config.get("min_child_weight_small", 3)
            logger.info(f"  Using small-scale XGB hyperparameters (N={n_samples})")
        else:
            n_est = self.xgb_config.get("n_estimators", 300)
            max_d = self.xgb_config.get("max_depth", 5)
            min_c = self.xgb_config.get("min_child_weight", 1)
            logger.info(f"  Using large-scale XGB hyperparameters (N={n_samples})")
            
        return xgb.XGBClassifier(
            n_estimators=n_est,
            max_depth=max_d,
            learning_rate=self.xgb_config.get("learning_rate", 0.05),
            subsample=self.xgb_config.get("subsample", 0.8),
            colsample_bytree=self.xgb_config.get("colsample_bytree", 0.8),
            min_child_weight=min_c,
            gamma=self.xgb_config.get("gamma", 0.5),
            reg_alpha=self.xgb_config.get("reg_alpha", 0.1),
            reg_lambda=self.xgb_config.get("reg_lambda", 2.0),
            eval_metric=self.xgb_config.get("eval_metric", "logloss"),
            random_state=self.xgb_config.get("random_state", 42),
            scale_pos_weight=scale_pos_weight,
            n_jobs=-1
        )

    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> None:
        """
        Trains binary classifiers for each persona.
        
        Args:
            X: Feature matrix from FeatureEngine.
            y: Boolean DataFrame with columns matching self.personas. 
               Index must match X.
        """
        # Filter out absolute features that cause cross-run overfitting
        absolute_cols = ["total_volume", "buy_volume", "sell_volume", "trade_count", "buy_trade_count", "sell_trade_count", "total_vol"]
        X_clean = X.drop(columns=[c for c in absolute_cols if c in X.columns])
        
        logger.info(f"Training supervised models on {len(X_clean)} samples for personas: {self.personas}")
        
        for persona in self.personas:
            if persona not in y.columns:
                logger.warning(f"Persona '{persona}' not found in labels DataFrame. Skipping.")
                continue
                
            y_target = y[persona].astype(int)
            
            # Dynamic scale_pos_weight for severe class imbalance
            n_pos = y_target.sum()
            n_neg = len(y_target) - n_pos
            
            if n_pos == 0 or n_neg == 0:
                logger.warning(f"[{persona}] Cannot train: {n_pos} positive, {n_neg} negative. Skipping.")
                continue
                
            scale_pos_weight = (n_neg / n_pos) if n_pos > 0 else 1.0
            n_samples = len(X_clean)
            
            logger.info(f"[{persona}] Class distribution: {n_pos} positive, {n_neg} negative (scale_pos_weight={scale_pos_weight:.2f})")
            
            if persona == "pump_and_dump":
                # XGBoost + StandardScaler on price features to prevent scale dominance
                X_scaled = X_clean.copy()
                scale_cols = [c for c in PUMP_DUMP_SCALE_FEATURES if c in X_scaled.columns]
                if scale_cols:
                    self._pd_scaler = StandardScaler()
                    X_scaled[scale_cols] = self._pd_scaler.fit_transform(X_scaled[scale_cols])
                model = self._make_xgb(scale_pos_weight, n_samples)
                model.fit(X_scaled, y_target)
                
            else:
                # Default: XGBoost for spoofing, layering, and wash_trading
                model = self._make_xgb(scale_pos_weight, n_samples)
                model.fit(X_clean, y_target)
                
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
        
        # Filter out absolute features that cause cross-run overfitting
        absolute_cols = ["total_volume", "buy_volume", "sell_volume", "trade_count", "buy_trade_count", "sell_trade_count", "total_vol"]
        X_clean = X.drop(columns=[c for c in absolute_cols if c in X.columns])

        for persona, model in self.models.items():
            X_input = X_clean.copy()
            
            # Apply pump_and_dump scaler if it was fitted
            if persona == "pump_and_dump" and self._pd_scaler is not None:
                scale_cols = [c for c in PUMP_DUMP_SCALE_FEATURES if c in X_input.columns]
                if scale_cols:
                    X_input[scale_cols] = self._pd_scaler.transform(X_input[scale_cols])
            
            # predict_proba returns [P(class=0), P(class=1)]
            probs = model.predict_proba(X_input)
            results[persona] = probs[:, 1]

        return results


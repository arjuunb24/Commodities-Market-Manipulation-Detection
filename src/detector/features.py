"""
src/detector/features.py
========================
Feature Engineering for the Veridex Detector Subsystem (Phase 3).

Converts raw tick-level order_log and trade_log data into a matrix of rolling 
window statistical features used by the Machine Learning models.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

class FeatureEngine:
    """
    Extracts time-series features from raw ABM logs.
    """
    
    def __init__(self, config: dict[str, Any]) -> None:
        """
        Args:
            config: Detector config mapping (loaded from detector_config.yaml)
        """
        self.window = config.get("feature_engine", {}).get("window", "5min")
        self.step = config.get("feature_engine", {}).get("step", "1min")
        self.top_k = config.get("feature_engine", {}).get("top_k_traders", 5)

    def generate_features(self, trade_df: pd.DataFrame, order_df: pd.DataFrame) -> pd.DataFrame:
        """
        Generates rolling window features per commodity.
        Returns a DataFrame indexed by (commodity, window_end_time).
        """
        logger.info(f"Generating features with window={self.window}, step={self.step}...")
        
        # 1. Map ticks to a synthetic Datetime index (1 tick = 1 second)
        # This allows us to use Pandas native time-based rolling/resample logic.
        t_df = trade_df.copy()
        o_df = order_df.copy()
        
        base_time = pd.Timestamp("2026-01-01 00:00:00")
        if not t_df.empty:
            t_df["timestamp"] = base_time + pd.to_timedelta(t_df["tick"], unit="s")
        if not o_df.empty:
            o_df["timestamp"] = base_time + pd.to_timedelta(o_df["tick"], unit="s")

        # 2. Process per commodity to avoid bleeding states across assets
        commodities = set(t_df["commodity"].unique() if not t_df.empty else []) | set(o_df["commodity"].unique() if not o_df.empty else [])
        
        all_features = []
        for comm in commodities:
            comm_trades = t_df[t_df["commodity"] == comm] if not t_df.empty else pd.DataFrame()
            comm_orders = o_df[o_df["commodity"] == comm] if not o_df.empty else pd.DataFrame()
            
            df_feat = self._process_single_commodity(comm_trades, comm_orders)
            df_feat["commodity"] = comm
            all_features.append(df_feat)
            
        if not all_features:
            return pd.DataFrame()
            
        final_df = pd.concat(all_features).reset_index()
        # Set a multi-index for cleaner ML processing
        final_df = final_df.set_index(["commodity", "timestamp"]).sort_index()
        return final_df

    def _process_single_commodity(self, trades: pd.DataFrame, orders: pd.DataFrame) -> pd.DataFrame:
        """Extracts features for a single commodity over the rolling window."""
        
        # Create a unified timeline by resampling both logs to the step size, then rolling.
        # We will resample to the `step` frequency, then apply a rolling window of `window`.
        
        # --- Orders Aggregation ---
        if not orders.empty:
            orders = orders.set_index("timestamp").sort_index()
            # Count SUBMIT vs CANCEL
            o_resampled = orders.assign(
                is_submit=(orders["event_type"] == "SUBMIT").astype(int),
                is_cancel=(orders["event_type"] == "CANCEL").astype(int)
            ).resample(self.step).agg({
                "is_submit": "sum",
                "is_cancel": "sum",
            })
        else:
            o_resampled = pd.DataFrame(columns=["is_submit", "is_cancel"])
            
        # --- Trades Aggregation ---
        if not trades.empty:
            trades = trades.set_index("timestamp").sort_index()
            
            t_resampled = trades.assign(
                buy_vol=np.where(~trades["buyer_trader_id"].str.startswith("mm"), trades["quantity"], 0),
                sell_vol=np.where(~trades["seller_trader_id"].str.startswith("mm"), trades["quantity"], 0),
                total_vol=trades["quantity"]
            ).resample(self.step).agg({
                "total_vol": "sum",
                "buy_vol": "sum",
                "sell_vol": "sum",
                "trade_id": "count"
            }).rename(columns={"trade_id": "trade_count"})
            
            # Custom volume concentration (Top K)
            def _top_k_concentration(grp):
                if grp.empty or grp["quantity"].sum() == 0:
                    return 0.0
                trader_vols = pd.concat([
                    grp.groupby("buyer_trader_id")["quantity"].sum(),
                    grp.groupby("seller_trader_id")["quantity"].sum()
                ])
                trader_vols = trader_vols.groupby(trader_vols.index).sum()
                top_k_vol = trader_vols.nlargest(self.top_k).sum()
                return top_k_vol / (grp["quantity"].sum() * 2)
                
            conc = trades.resample(self.step).apply(_top_k_concentration)
            if isinstance(conc, pd.DataFrame):
                 conc = conc.iloc[:, 0] if not conc.empty else pd.Series(dtype=float)
            conc.name = "volume_concentration"
            t_resampled = t_resampled.join(conc)
        else:
            t_resampled = pd.DataFrame(columns=["total_vol", "buy_vol", "sell_vol", "trade_count", "volume_concentration"])
            
        # --- Combine and Roll ---
        combined = pd.concat([o_resampled, t_resampled], axis=1).fillna(0)
        rolled = combined.rolling(window=self.window, min_periods=1).sum()
        
        if "volume_concentration" in combined.columns:
            rolled["volume_concentration"] = combined["volume_concentration"].rolling(window=self.window, min_periods=1).mean()
        
        # --- Compute Final Derived Features ---
        features = pd.DataFrame(index=rolled.index)
        
        total_orders = rolled["is_submit"] + rolled["is_cancel"]
        features["cancel_rate"] = np.where(total_orders > 0, rolled["is_cancel"] / total_orders, 0.0)
        
        features["otr"] = np.where(rolled["trade_count"] > 0, rolled["is_submit"] / rolled["trade_count"], rolled["is_submit"])
        
        features["trade_imbalance"] = np.where(rolled["total_vol"] > 0, 
                                               (rolled["buy_vol"] - rolled["sell_vol"]) / rolled["total_vol"], 
                                               0.0)
                                               
        features["volume_concentration"] = rolled.get("volume_concentration", 0.0)
        features["total_volume"] = rolled.get("total_vol", 0.0)
        
        features = features.fillna(0.0)
        
        return features

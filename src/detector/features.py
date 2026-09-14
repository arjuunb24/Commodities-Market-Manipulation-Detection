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
        # Number of historical windows used for volume_spike_ratio baseline
        self.vol_spike_lookback = config.get("feature_engine", {}).get("vol_spike_lookback", 10)

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
            # Count placed vs cancelled and Volume
            o_resampled = orders.assign(
                is_submit=(orders["event_type"] == "placed").astype(int),
                is_cancel=(orders["event_type"] == "cancelled").astype(int),
                submit_vol=np.where(orders["event_type"] == "placed", orders["quantity"], 0.0),
                cancel_vol=np.where(orders["event_type"] == "cancelled", orders["quantity"], 0.0)
            ).resample(self.step).agg({
                "is_submit": "sum",
                "is_cancel": "sum",
                "submit_vol": "sum",
                "cancel_vol": "sum"
            })
        else:
            o_resampled = pd.DataFrame(columns=["is_submit", "is_cancel", "submit_vol", "cancel_vol"])
            
        # --- Trades Aggregation ---
        if not trades.empty:
            trades = trades.set_index("timestamp").sort_index()
            
            trades_ext = trades.assign(
                buy_vol=np.where(~trades["buyer_trader_id"].str.startswith("mm"), trades["quantity"], 0),
                sell_vol=np.where(~trades["seller_trader_id"].str.startswith("mm"), trades["quantity"], 0),
                total_vol=trades["quantity"],
                trade_val=trades["price"] * trades["quantity"]
            )
            
            t_resampled = trades_ext.resample(self.step).agg({
                "total_vol": "sum",
                "buy_vol": "sum",
                "sell_vol": "sum",
                "trade_id": "count",
                "trade_val": "sum"
            }).rename(columns={"trade_id": "trade_count"})
            
            t_resampled["vwap"] = np.where(t_resampled["total_vol"] > 0, t_resampled["trade_val"] / t_resampled["total_vol"], np.nan)
            t_resampled["vwap"] = t_resampled["vwap"].ffill()

            # --- Custom volume concentration (Top K) ---
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

            # --- WASH TRADING Feature 1: Top-K Circular Volume Ratio ---
            # KEY FIX: Wash traders are only ~5% of total market volume.
            # We identify the TOP-K traders by volume first, then compute
            # circular volume ratio ONLY on those high-volume traders.
            # A wash pair will dominate among the top-K active traders,
            # and their circular volume will be ~1.0 (perfect round trips).
            def _topk_circular_ratio(grp, k=5):
                if grp.empty or grp["quantity"].sum() == 0:
                    return 0.0
                # Get the K traders with highest combined (buy+sell) volume
                buys = grp.groupby("buyer_trader_id")["quantity"].sum()
                sells = grp.groupby("seller_trader_id")["quantity"].sum()
                combined = pd.concat([buys, sells]).groupby(level=0).sum()
                top_traders = combined.nlargest(k).index
                
                # Restrict to trades involving top-K traders
                top_grp = grp[grp["buyer_trader_id"].isin(top_traders) | grp["seller_trader_id"].isin(top_traders)]
                if top_grp.empty:
                    return 0.0
                
                top_buys = top_grp.groupby("buyer_trader_id")["quantity"].sum()
                top_sells = top_grp.groupby("seller_trader_id")["quantity"].sum()
                common = top_buys.index.intersection(top_sells.index)
                if common.empty:
                    return 0.0
                circ_vol = pd.DataFrame({"buys": top_buys[common], "sells": top_sells[common]}).min(axis=1).sum()
                return circ_vol / max(top_grp["quantity"].sum(), 1)
                
            circ_ratio = trades.resample(self.step).apply(_topk_circular_ratio)
            if isinstance(circ_ratio, pd.DataFrame):
                 circ_ratio = circ_ratio.iloc[:, 0] if not circ_ratio.empty else pd.Series(dtype=float)
            circ_ratio.name = "circular_volume_ratio"
            t_resampled = t_resampled.join(circ_ratio)

            # --- WASH TRADING Feature 2: Min Net Position Ratio (Top-K) ---
            # KEY FIX: Same top-K strategy. For the K most active traders,
            # find the MINIMUM (most round-trip) net position ratio.
            # Wash traders have net_pos ≈ 0; legitimate traders are directional (net_pos ≈ 1).
            # A single wash pair in the top-K will drive this feature to near 0.
            def _min_net_position_topk(grp, k=8):
                if grp.empty or grp["quantity"].sum() == 0:
                    return 1.0
                buys = grp.groupby("buyer_trader_id")["quantity"].sum()
                sells = grp.groupby("seller_trader_id")["quantity"].sum()
                all_traders = buys.index.union(sells.index)
                # Focus on top-K by total volume
                combined = pd.concat([buys, sells]).groupby(level=0).sum()
                top_traders = combined.nlargest(k).index
                
                ratios = []
                for trader in top_traders:
                    b = float(buys.get(trader, 0.0))
                    s = float(sells.get(trader, 0.0))
                    total = b + s
                    if total > 0:
                        ratios.append(abs(b - s) / total)  # 0 = wash, 1 = directional
                return float(min(ratios)) if ratios else 1.0  # min = worst offender
                
            net_pos = trades.resample(self.step).apply(_min_net_position_topk)
            if isinstance(net_pos, pd.DataFrame):
                 net_pos = net_pos.iloc[:, 0] if not net_pos.empty else pd.Series(dtype=float)
            net_pos.name = "net_position_ratio"
            t_resampled = t_resampled.join(net_pos)

            # --- WASH TRADING Feature 3: Repeated Pair Concentration ---
            # KEY FIX: Instead of unique_pairs/total_trades market-wide,
            # we find the SINGLE most repeated trading pair and compute
            # what fraction of total volume they represent. A wash pair
            # trading 1,000 times together is a massive red flag.
            def _top_pair_concentration(grp):
                if grp.empty or len(grp) == 0:
                    return 0.0
                pair_vol = grp.groupby(["buyer_trader_id", "seller_trader_id"])["quantity"].sum()
                if pair_vol.empty:
                    return 0.0
                total_vol = grp["quantity"].sum()
                return float(pair_vol.max() / total_vol) if total_vol > 0 else 0.0
                
            cp_ratio = trades.resample(self.step).apply(_top_pair_concentration)
            if isinstance(cp_ratio, pd.DataFrame):
                 cp_ratio = cp_ratio.iloc[:, 0] if not cp_ratio.empty else pd.Series(dtype=float)
            cp_ratio.name = "unique_counterparty_ratio"
            t_resampled = t_resampled.join(cp_ratio)

            # --- WASH TRADING Feature 4: Trade Size Uniformity (Coefficient of Variation) ---
            # Automated wash trading scripts use fixed round sizes. Low CV = suspicious.
            def _trade_size_cv(grp):
                if grp.empty or len(grp) < 2:
                    return 1.0
                mean_q = grp["quantity"].mean()
                if mean_q == 0:
                    return 1.0
                return grp["quantity"].std() / mean_q
                
            size_cv = trades.resample(self.step).apply(_trade_size_cv)
            if isinstance(size_cv, pd.DataFrame):
                 size_cv = size_cv.iloc[:, 0] if not size_cv.empty else pd.Series(dtype=float)
            size_cv.name = "trade_size_cv"

            t_resampled = t_resampled.join(size_cv)
        else:
            t_resampled = pd.DataFrame(columns=["total_vol", "buy_vol", "sell_vol", "trade_count", "volume_concentration"])
            
        # --- Combine and Roll ---
        combined = pd.concat([o_resampled, t_resampled], axis=1).fillna(0)
        
        if "vwap" in t_resampled.columns:
            combined["vwap"] = t_resampled["vwap"].ffill()
            
        rolled = combined.rolling(window=self.window, min_periods=1).sum()
        
        # These features are ratios so they should be averaged, not summed
        for ratio_col in ["volume_concentration", "circular_volume_ratio",
                          "net_position_ratio", "unique_counterparty_ratio", "trade_size_cv"]:
            if ratio_col in combined.columns:
                rolled[ratio_col] = combined[ratio_col].rolling(window=self.window, min_periods=1).mean()
        
        # --- Compute Final Derived Features ---
        features = pd.DataFrame(index=rolled.index)
        
        # --- Generic / Spoofing Features ---
        total_orders = rolled["is_submit"] + rolled["is_cancel"]
        features["cancel_rate"] = np.where(total_orders > 0, rolled["is_cancel"] / total_orders, 0.0)
        features["otr"] = np.where(rolled["trade_count"] > 0, rolled["is_submit"] / rolled["trade_count"], rolled["is_submit"])
        features["trade_imbalance"] = np.where(rolled["total_vol"] > 0, 
                                               (rolled["buy_vol"] - rolled["sell_vol"]) / rolled["total_vol"], 
                                               0.0)
        features["volume_concentration"] = rolled.get("volume_concentration", 0.0)
        total_order_vol = rolled.get("submit_vol", 0.0) + rolled.get("cancel_vol", 0.0)
        features["cancel_volume_ratio"] = np.where(total_order_vol > 0, rolled.get("cancel_vol", 0.0) / total_order_vol, 0.0)
        features["avg_trade_size"] = np.where(rolled["trade_count"] > 0, rolled["total_vol"] / rolled["trade_count"], 0.0)
        features["total_volume"] = rolled.get("total_vol", 0.0)
        
        # NEW RATIO FEATURES: To capture Spoofing impact without absolute scale dependency
        # 1. order_rate_ratio: Does the number of orders spike? (Spoofing creates massive order floods)
        order_series = total_orders
        rolling_mean_orders = order_series.rolling(window=self.vol_spike_lookback, min_periods=1).mean().shift(1)
        rolling_mean_orders = rolling_mean_orders.fillna(order_series).replace(0, 1.0)
        features["order_rate_ratio"] = np.where(rolling_mean_orders > 0, order_series / rolling_mean_orders, 1.0)
        
        # 2. trade_rate_ratio: Does the number of actual trades drop? (Spoofing dries up real liquidity)
        trade_series = rolled["trade_count"]
        rolling_mean_trades = trade_series.rolling(window=self.vol_spike_lookback, min_periods=1).mean().shift(1)
        rolling_mean_trades = rolling_mean_trades.fillna(trade_series).replace(0, 1.0)
        features["trade_rate_ratio"] = np.where(rolling_mean_trades > 0, trade_series / rolling_mean_trades, 1.0)
        
        # 3. Average Order Size (The true Spoofing signal): Spoofers place massive orders (e.g. 8x best bid)
        features["avg_submit_size"] = np.where(rolled["is_submit"] > 0, rolled["submit_vol"] / rolled["is_submit"], 0.0)
        features["avg_cancel_size"] = np.where(rolled["is_cancel"] > 0, rolled["cancel_vol"] / rolled["is_cancel"], 0.0)
        
        # 4. Top-K Max Cancel Ratio: Market-wide cancel_rate is diluted by MM orders. 
        # Find the Top-K most active order placers and take their max cancel_rate.
        if not orders.empty:
            def _max_cancel_ratio_topk(grp, k=5):
                if grp.empty or grp["quantity"].sum() == 0:
                    return 0.0
                submits = grp[grp["event_type"] == "placed"].groupby("trader_id")["quantity"].sum()
                cancels = grp[grp["event_type"] == "cancelled"].groupby("trader_id")["quantity"].sum()
                if submits.empty:
                    return 0.0
                top_traders = submits.nlargest(k).index
                ratios = []
                for trader in top_traders:
                    s = float(submits.get(trader, 0.0))
                    c = float(cancels.get(trader, 0.0))
                    if s > 0:
                        ratios.append(c / s)
                return max(ratios) if ratios else 0.0
            
            topk_cancel = orders.resample(self.step).apply(_max_cancel_ratio_topk)
            if isinstance(topk_cancel, pd.DataFrame):
                topk_cancel = topk_cancel.iloc[:, 0] if not topk_cancel.empty else pd.Series(dtype=float)
            topk_cancel.name = "max_cancel_ratio_topk"
            topk_cancel_rolled = topk_cancel.rolling(window=self.window, min_periods=1).mean()
            features["max_cancel_ratio_topk"] = topk_cancel_rolled.reindex(features.index).fillna(0.0)
        else:
            features["max_cancel_ratio_topk"] = 0.0

        # --- Wash Trading Features ---
        features["circular_volume_ratio"] = rolled.get("circular_volume_ratio", 0.0)
        features["net_position_ratio"] = rolled.get("net_position_ratio", 1.0)
        features["unique_counterparty_ratio"] = rolled.get("unique_counterparty_ratio", 1.0)
        features["trade_size_cv"] = rolled.get("trade_size_cv", 1.0)
        
        # --- Pump & Dump Features ---
        if "vwap" in combined.columns:
            vwap = combined["vwap"]
            
            min_vwap = vwap.rolling(window=self.window, min_periods=1).min()
            max_vwap = vwap.rolling(window=self.window, min_periods=1).max()
            mean_vwap = vwap.rolling(window=self.window, min_periods=1).mean()
            std_vwap = vwap.rolling(window=self.window, min_periods=1).std().fillna(0)
            
            # Price range and volatility
            features["price_momentum"] = np.where(min_vwap > 0, (vwap - min_vwap) / min_vwap, 0.0)
            features["price_range"] = np.where(min_vwap > 0, (max_vwap - min_vwap) / min_vwap, 0.0)
            features["price_volatility"] = np.where(mean_vwap > 0, std_vwap / mean_vwap, 0.0)
            
            # price_return: how much did price move vs previous window (multi-period directional signal)
            prev_vwap = vwap.shift(1).ffill()
            features["price_return"] = np.where(prev_vwap > 0, (vwap - prev_vwap) / prev_vwap, 0.0)
            
            # volume_spike_ratio: current volume vs rolling historical average (gold standard P&D signal)
            vol_series = combined.get("total_vol", pd.Series(0.0, index=combined.index))
            # FIX: If we have no history, the rolling mean is NaN. If we fillna with 1.0, 
            # and current volume is 80,000, the spike ratio becomes 80,000 for the first window!
            # Instead, fillna with the current volume so the initial ratio is 1.0 (no spike).
            rolling_mean_vol = vol_series.rolling(window=self.vol_spike_lookback, min_periods=1).mean().shift(1)
            rolling_mean_vol = rolling_mean_vol.fillna(vol_series).replace(0, 1.0)
            features["volume_spike_ratio"] = np.where(rolling_mean_vol > 0, vol_series / rolling_mean_vol, 1.0)
            
            # abnormal_return_z: Z-score of (volume * abs_price_change) — academic gold-standard P&D signal
            abs_return = (vwap - prev_vwap).abs()
            abnormal_return = vol_series * abs_return
            ar_mean = abnormal_return.rolling(window=self.vol_spike_lookback, min_periods=1).mean()
            ar_std = abnormal_return.rolling(window=self.vol_spike_lookback, min_periods=1).std().fillna(1.0).replace(0, 1.0)
            features["abnormal_return_z"] = (abnormal_return - ar_mean) / ar_std

            # --- PUMP & DUMP Feature: Top-K Buy Imbalance ---
            # KEY FIX: PumpDump accounts only represent 3.5% of total volume.
            # Instead of computing market-wide buy/sell imbalance (which averages to ~0),
            # we look at the TOP-K most active BUYERS and compute what fraction of their
            # activity is buys. During a pump, the top-K buyers are buying 95%+ of the time.
            # This pinpoints the aggressive accumulation phase despite the noise.
            def _topk_buy_imbalance(grp, k=5):
                if grp.empty or grp["quantity"].sum() == 0:
                    return 0.0
                # Find the K most active buyers by volume
                top_buyers = grp.groupby("buyer_trader_id")["quantity"].sum().nlargest(k).index
                
                # Among trades involving those buyers as buyer
                buy_trades = grp[grp["buyer_trader_id"].isin(top_buyers)]
                if buy_trades.empty:
                    return 0.0
                buy_vol = buy_trades["quantity"].sum()
                
                # How much do those same traders sell?
                sell_vol = grp[grp["seller_trader_id"].isin(top_buyers)]["quantity"].sum()
                
                total = buy_vol + sell_vol
                return buy_vol / total if total > 0 else 0.5  # 1.0 = pure buyer (pumping)
                
            topk_buy_imbal = trades.resample(self.step).apply(_topk_buy_imbalance)
            if isinstance(topk_buy_imbal, pd.DataFrame):
                topk_buy_imbal = topk_buy_imbal.iloc[:, 0] if not topk_buy_imbal.empty else pd.Series(dtype=float)
            topk_buy_imbal.name = "topk_buy_imbalance"
            topk_buy_imbal_rolled = topk_buy_imbal.rolling(window=self.window, min_periods=1).mean()
            features["topk_buy_imbalance"] = topk_buy_imbal_rolled.reindex(features.index).fillna(0.5)
        else:
            features["price_momentum"] = 0.0
            features["price_range"] = 0.0
            features["price_volatility"] = 0.0
            features["price_return"] = 0.0
            features["volume_spike_ratio"] = 1.0
            features["abnormal_return_z"] = 0.0
            features["topk_buy_imbalance"] = 0.5
        
        features = features.fillna(0.0)
        
        return features


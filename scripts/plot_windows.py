"""
scripts/plot_windows.py
Visualizes manipulation events alongside multi-scale rolling window features.
"""

import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
import yaml

from src.detector.features import FeatureEngine

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/runs/adversarial_20260915T163319Z_seed42/round_0/test")
    parser.add_argument("--out", type=str, default="feature_analysis.png")
    args = parser.parse_args()

    print(f"Loading data from {args.data_dir}...")
    trades = pd.read_parquet(os.path.join(args.data_dir, "trade_log.parquet"))
    orders = pd.read_parquet(os.path.join(args.data_dir, "order_log.parquet"))
    events = pd.read_parquet(os.path.join(args.data_dir, "manipulation_events.parquet"))
    
    # Load config to get window sizes
    with open("configs/detector_config.yaml", "r") as f:
        config = yaml.safe_load(f)

    print("Generating features...")
    fe = FeatureEngine(config)
    market, _ = fe.compute(trades, orders)

    # Extract market-level features for the first commodity, indexed by window end time
    commodity = market.index.get_level_values("commodity").unique()[0]
    df = market.xs(commodity, level="commodity").copy()
    df.index = pd.Timestamp("2026-01-01 00:00:00") + pd.to_timedelta(df["window_end"], unit="s")
    
    # Get VWAP from the resampled dataframe (if possible, or just trades)
    base_time = pd.Timestamp("2026-01-01 00:00:00")
    trades["timestamp"] = base_time + pd.to_timedelta(trades["tick"], unit="s")
    trades_comm = trades[trades["commodity"] == commodity].set_index("timestamp").sort_index()
    
    trades_comm["trade_val"] = trades_comm["price"] * trades_comm["quantity"]
    vwap = trades_comm.resample("10s").agg({"trade_val": "sum", "quantity": "sum"})
    vwap["vwap"] = vwap["trade_val"] / vwap["quantity"]
    vwap["vwap"] = vwap["vwap"].ffill()

    # Create figure
    fig, axes = plt.subplots(3, 1, figsize=(15, 12), sharex=True)
    
    # 1. Price Chart with Manipulation Shading
    ax = axes[0]
    ax.plot(vwap.index, vwap["vwap"], color="black", label="VWAP", linewidth=1)
    ax.set_title(f"Commodity: {commodity} - Price and Manipulation Events")
    ax.set_ylabel("Price")
    
    colors = {"spoofing": "red", "pump_and_dump": "orange", "wash_trading": "purple"}
    
    if not events.empty:
        comm_events = events[events["commodity"] == commodity]
        for _, row in comm_events.iterrows():
            start = base_time + pd.to_timedelta(row["start_tick"], unit="s")
            end = base_time + pd.to_timedelta(row["end_tick"], unit="s")
            persona = row["persona"]
            color = colors.get(persona, "grey")
            ax.axvspan(start, end, color=color, alpha=0.3, label=f"Event: {persona}")
            
    # Remove duplicate labels in legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc="upper left")
    
    # 2. Order-flow pressure (market level)
    ax = axes[1]
    ax.plot(df.index, df["mkt_cancel_ratio"], color="red", label="cancel ratio", linewidth=1.5)
    ax.plot(df.index, df["mkt_order_imbalance"], color="blue", label="placed-volume imbalance", linewidth=1.2, alpha=0.7)
    ax.set_title("Order-flow features (market level)")
    ax.legend(loc="upper left")

    # 3. Volume / return shocks (market level)
    ax = axes[2]
    ax.plot(df.index, df["mkt_volume_spike"], color="orange", label="volume spike ratio", linewidth=1.5)
    ax.plot(df.index, df["mkt_abnormal_return_z"], color="green", label="abnormal return z", linewidth=1.2, alpha=0.7)
    ax.set_title("Volume and return shocks (market level)")
    ax.legend(loc="upper left")

    plt.tight_layout()
    plt.savefig(args.out)
    print(f"Plot saved to {args.out}")

if __name__ == "__main__":
    main()

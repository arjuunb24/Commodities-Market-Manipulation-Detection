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
    features = fe.generate_features(trades, orders)
    
    # Extract data for the first commodity
    commodity = features.index.get_level_values("commodity").unique()[0]
    df = features.xs(commodity, level="commodity").copy()
    
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
    
    # 2. Spoofing Features (Short vs Long Window)
    ax = axes[1]
    if "cancel_rate_30s" in df.columns:
        ax.plot(df.index, df["cancel_rate_30s"], color="red", label="cancel_rate_30s", linewidth=1.5)
    if "cancel_rate_15min" in df.columns:
        ax.plot(df.index, df["cancel_rate_15min"], color="blue", label="cancel_rate_15min", linewidth=1.5, alpha=0.6)
    
    ax.set_title("Spoofing Features (Cancel Rate) - Short vs Long Window")
    ax.set_ylabel("Cancel Rate")
    ax.legend(loc="upper left")
    
    # 3. Pump & Dump Features (Volume Spike or Imbalance)
    ax = axes[2]
    if "abnormal_return_z_15min" in df.columns:
        ax.plot(df.index, df["abnormal_return_z_15min"], color="orange", label="abnormal_return_z_15min", linewidth=1.5)
    if "abnormal_return_z_60min" in df.columns:
        ax.plot(df.index, df["abnormal_return_z_60min"], color="green", label="abnormal_return_z_60min", linewidth=1.5, alpha=0.6)
        
    ax.set_title("Pump & Dump Features (Abnormal Return Z-Score) - 15m vs 60m")
    ax.set_ylabel("Z-Score")
    ax.legend(loc="upper left")
    
    plt.tight_layout()
    plt.savefig(args.out)
    print(f"Plot saved to {args.out}")

if __name__ == "__main__":
    main()

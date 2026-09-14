"""
scripts/run_simulation.py
==========================
CLI entry point for running a Veridex ABM simulation.

Loads configurations, runs the simulation, and prints a summary.
"""

import argparse
import sys
from pathlib import Path

import yaml

# Ensure src is in PYTHONPATH
sys.path.append(str(Path(__file__).parent.parent))

from src.core.calibration import CalibrationEngine
from src.core.simulation import SimConfig, run_simulation
from src.utils.logging_config import setup_logging
import pandas as pd
from datetime import datetime, timezone
from src.utils import io as io_utils

COMMODITIES = [
    "crude_oil_wti",
    "brent_crude",
    "gold",
    "silver",
    "natural_gas",
    "copper",
    "wheat",
    "corn"
]


def main() -> None:
    setup_logging()

    parser = argparse.ArgumentParser(description="Run Veridex ABM Simulation")
    parser.add_argument("--seed", type=int, default=42, help="Master random seed")
    parser.add_argument("--ticks", type=int, default=10000, help="Number of ticks to simulate")
    parser.add_argument("--outdir", type=str, default="data/runs", help="Output directory")
    parser.add_argument("--commodity", type=str, default="gold", help="Commodity to simulate (e.g. gold, crude_oil_wti) or 'all' for all 8.")
    
    args = parser.parse_args()

    engine = CalibrationEngine()
    
    if args.commodity.lower() == "all":
        coms_to_run = COMMODITIES
    else:
        coms_to_run = [args.commodity]

    all_trade_logs = []
    all_order_logs = []
    all_manip_logs = []
    all_bars_logs = []

    print(f"Running simulation for {len(coms_to_run)} commodities with seed {args.seed} for {args.ticks} ticks...")
    
    ts = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{ts}_{args.seed}"
    run_dir = io_utils.ensure_run_dir(Path(args.outdir), run_id, overwrite=True)

    for com in coms_to_run:
        print(f" -> Simulating {com}...")
        ou_target, _ = engine.get_fallback(com)

        config = SimConfig(
            commodity=com,
            fundamental_value_drift=ou_target.drift,
            fundamental_value_vol=ou_target.volatility,
            n_ticks=args.ticks,
            data_dir=Path(args.outdir),
            overwrite=True,
            write_to_disk=False  # Do not write to disk yet
        )

        # Apply LLM mutations from persona_config.yaml if they exist
        persona_config_path = Path(__file__).parent.parent / "configs" / "persona_config.yaml"
        if persona_config_path.exists():
            with open(persona_config_path) as f:
                p_cfg = yaml.safe_load(f) or {}

            # Spoofing mutations
            if "spoofing" in p_cfg:
                sp_cfg = p_cfg["spoofing"]
                if "cancellation_delay_ticks" in sp_cfg: config.spoof_cancel_delay = sp_cfg["cancellation_delay_ticks"]
                if "order_size_multiplier" in sp_cfg: config.spoof_size_multiplier = float(sp_cfg["order_size_multiplier"])
                if "frequency" in sp_cfg: config.spoof_frequency = float(sp_cfg["frequency"])
                if "price_aggressiveness" in sp_cfg: config.spoof_aggressiveness = float(sp_cfg["price_aggressiveness"])

            # Wash trading mutations
            if "wash_trading" in p_cfg:
                wt_cfg = p_cfg["wash_trading"]
                if "trade_frequency" in wt_cfg: config.wash_frequency = float(wt_cfg["trade_frequency"])
                if "price_deviation_from_mid" in wt_cfg: config.wash_price_deviation = float(wt_cfg["price_deviation_from_mid"])
                if "n_colluding_pairs" in wt_cfg: config.wash_n_pairs = int(wt_cfg["n_colluding_pairs"])

            # Pump and dump mutations
            if "pump_and_dump" in p_cfg:
                pnd_cfg = p_cfg["pump_and_dump"]
                if "burst_duration_ticks" in pnd_cfg: config.pnd_burst_duration = int(pnd_cfg["burst_duration_ticks"])
                if "n_coordinated_accounts" in pnd_cfg: config.pnd_n_accounts = int(pnd_cfg["n_coordinated_accounts"])
                if "dump_delay_ticks" in pnd_cfg: config.pnd_dump_delay = int(pnd_cfg["dump_delay_ticks"])
                if "accumulation_size" in pnd_cfg: config.pnd_accumulation_size = int(pnd_cfg["accumulation_size"])

            # Layering mutations
            if "layering" in p_cfg:
                ly_cfg = p_cfg["layering"]
                if "n_layers" in ly_cfg: config.layer_n_layers = int(ly_cfg["n_layers"])
                if "layer_spacing" in ly_cfg: config.layer_spacing = float(ly_cfg["layer_spacing"])
                if "cancellation_delay_ticks" in ly_cfg: config.layer_cancel_delay = int(ly_cfg["cancellation_delay_ticks"])

        output = run_simulation(config, args.seed)
        all_trade_logs.append(output.trade_log)
        all_order_logs.append(output.order_log)
        all_manip_logs.append(output.manipulation_events)
        all_bars_logs.append(output.ohlcv_bars)

    print("\nMerging datasets...")
    final_trade_df = pd.concat(all_trade_logs, ignore_index=True)
    final_order_df = pd.concat(all_order_logs, ignore_index=True)
    final_manip_df = pd.concat(all_manip_logs, ignore_index=True)
    final_bars_df = pd.concat(all_bars_logs, ignore_index=True)

    print(f"Writing concatenated output to {run_dir}...")
    io_utils.write_parquet(final_trade_df, run_dir / "trade_log.parquet", "trade_log", overwrite=True)
    io_utils.write_parquet(final_order_df, run_dir / "order_log.parquet", "order_log", overwrite=True)
    io_utils.write_parquet(final_manip_df, run_dir / "manipulation_events.parquet", "manipulation_events", overwrite=True)
    io_utils.write_parquet(final_bars_df, run_dir / "ohlcv_bars.parquet", "ohlcv_bars", overwrite=True)

    # Write run metadata
    metadata = {
        "run_id": run_id,
        "seed": args.seed,
        "n_ticks": args.ticks,
        "commodities": coms_to_run
    }
    io_utils.write_yaml(metadata, run_dir / "run_metadata.yaml")
    
    print("\nSimulation complete!")
    print(f"Run ID: {run_id}")
    print(f"Output directory: {run_dir}")
    print(f"Trades logged: {len(final_trade_df)}")
    print(f"OHLCV bars generated: {len(final_bars_df)}")
    print(f"Manipulation episodes: {len(final_manip_df)}")


if __name__ == "__main__":
    main()

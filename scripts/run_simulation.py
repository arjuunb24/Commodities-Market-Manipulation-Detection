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
from src.core.simulation import SimConfig, multi_commodity_seed, run_simulation
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


def simulate_commodity_worker(com: str, seed: int, ticks: int, outdir: str, p_cfg: dict, tick_offset: int = 0) -> tuple:
    from src.core.calibration import CalibrationEngine
    from src.core.simulation import SimConfig, run_simulation
    from pathlib import Path
    
    engine = CalibrationEngine()
    ou_target, _ = engine.get_fallback(com)

    config = SimConfig(
        commodity=com,
        fundamental_value_drift=ou_target.drift,
        fundamental_value_vol=ou_target.volatility,
        n_ticks=ticks,
        tick_offset=tick_offset,
        data_dir=Path(outdir),
        overwrite=True,
        write_to_disk=False  # Do not write to disk yet
    )

    # Resolve persona configuration (round_0 baseline + mutations) via the shared mapping.
    from src.core.persona_params import apply_persona_config, resolve_persona_config
    apply_persona_config(config, resolve_persona_config(p_cfg))

    output = run_simulation(config, seed)
    return (com, output.trade_log, output.order_log, output.manipulation_events, output.ohlcv_bars)


def main() -> None:
    setup_logging()

    parser = argparse.ArgumentParser(description="Run Veridex ABM Simulation")
    parser.add_argument("--seed", type=int, default=42, help="Master random seed")
    parser.add_argument("--ticks", type=int, default=10000, help="Number of ticks to simulate")
    parser.add_argument("--outdir", type=str, default="data/runs", help="Output directory")
    parser.add_argument("--commodity", type=str, default="gold", help="Commodity to simulate (e.g. gold, crude_oil_wti) or 'all' for all 8.")
    parser.add_argument("--tick-offset", type=int, default=0, help="Tick offset for uninterrupted multi-round datasets")
    
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

    print(f"Running simulation for {len(coms_to_run)} commodities with seed {args.seed} for {args.ticks} ticks (tick_offset={args.tick_offset})...")
    
    ts = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{ts}_{args.seed}"
    run_dir = io_utils.ensure_run_dir(Path(args.outdir), run_id, overwrite=True)

    persona_config_path = Path(__file__).parent.parent / "configs" / "persona_config.yaml"
    p_cfg = {}
    if persona_config_path.exists():
        with open(persona_config_path) as f:
            p_cfg = yaml.safe_load(f) or {}

    from concurrent.futures import ProcessPoolExecutor, as_completed
    with ProcessPoolExecutor() as executor:
        futures = {}
        for com in coms_to_run:
            # If multiple commodities are run, decouple seeds to avoid identical price trajectories
            com_seed = multi_commodity_seed(args.seed, com) if len(coms_to_run) > 1 else args.seed
            futures[executor.submit(simulate_commodity_worker, com, com_seed, args.ticks, args.outdir, p_cfg, args.tick_offset)] = com
        
        for future in as_completed(futures):
            com = futures[future]
            try:
                _, trade_log, order_log, manip_log, bars_log = future.result()
                print(f" -> Finished {com}")
                all_trade_logs.append(trade_log)
                all_order_logs.append(order_log)
                all_manip_logs.append(manip_log)
                all_bars_logs.append(bars_log)
            except Exception as e:
                print(f" -> Error simulating {com}: {e}")

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

"""
scripts/run_full_pipeline.py
=============================
Phase 5: The Full Adversarial Loop.

Ties together:
    Round 0 (baseline)  → Simulate → Detect → Log metrics
    Round N (N > 0)     → LLM reads Round N-1 metrics → Mutates persona_config.yaml
                        → Simulate → Detect → Log metrics

Usage:
    .venv\\Scripts\\python.exe scripts/run_full_pipeline.py
    .venv\\Scripts\\python.exe scripts/run_full_pipeline.py --rounds 3 --ticks 2000 --seed 42
"""

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.adversarial.orchestrator import RoundOrchestrator
from src.adversarial.metrics import build_summary_table
from src.agentic.adversarial_strategist import AdversarialStrategist
from src.utils.logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


def print_metrics_table(summary_df):
    """Prints a clean per-round, per-persona metrics table to the console."""
    if summary_df.empty:
        return

    print("\n" + "=" * 72)
    print(f"{'Round':<8} {'Persona':<20} {'Precision':>10} {'Recall':>8} {'F1':>8} {'FPR':>8}")
    print("-" * 72)

    for _, row in summary_df.iterrows():
        # Highlight round boundaries
        print(
            f"{int(row['round']):<8} {row['persona']:<20} "
            f"{row['precision']:>10.3f} {row['recall']:>8.3f} "
            f"{row['f1']:>8.3f} {row['fpr']:>8.3f}"
        )

    print("=" * 72 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Veridex Full Adversarial Pipeline")
    parser.add_argument("--rounds",    type=int, default=3,               help="Number of adversarial rounds (including baseline Round 0)")
    parser.add_argument("--ticks",     type=int, default=5000,            help="Simulation ticks per round")
    parser.add_argument("--seed",      type=int, default=42,              help="Random seed for simulation")
    parser.add_argument("--commodity", type=str, default="crude_oil_wti", help="Commodity to simulate")
    parser.add_argument("--persona",   type=str, default="spoofing",      help="Persona to target for adversarial mutation (or 'all' for all enabled personas)")
    args = parser.parse_args()

    # ----------------------------------------------------------------
    # Setup
    # ----------------------------------------------------------------
    ts = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"adversarial_{ts}_seed{args.seed}"
    master_run_dir = REPO_ROOT / "data" / "runs" / run_id

    logger.info("=" * 60)
    logger.info(f"Veridex Adversarial Pipeline — Run ID: {run_id}")
    logger.info(f"Rounds: {args.rounds} | Ticks/Round: {args.ticks} | Seed: {args.seed}")
    logger.info(f"Commodity: {args.commodity} | Targeted Persona: {args.persona}")
    logger.info("=" * 60)

    # Load detector config
    detector_config_path = REPO_ROOT / "configs" / "detector_config.yaml"
    with open(detector_config_path) as f:
        detector_config = yaml.safe_load(f)

    # Load LLM config for Adversarial Strategist
    llm_config_path = REPO_ROOT / "configs" / "llm_config.yaml"
    with open(llm_config_path) as f:
        llm_config = yaml.safe_load(f)
    strategist_config = llm_config.get("adversarial_strategist", {})

    # Reset persona_config.yaml to baseline (remove top-level mutations)
    persona_config_path = REPO_ROOT / "configs" / "persona_config.yaml"
    if persona_config_path.exists():
        with open(persona_config_path) as f:
            p_cfg = yaml.safe_load(f) or {}
        # Keep only round_0 baseline
        baseline = {"round_0": p_cfg.get("round_0", {})}
        with open(persona_config_path, "w") as f:
            yaml.dump(baseline, f, default_flow_style=False)
        logger.info("Reset persona_config.yaml to baseline for Round 0.")

    # Initialise orchestrator (runs the simulation + detector each round)
    orchestrator = RoundOrchestrator(

        master_run_dir=master_run_dir,
        detector_config=detector_config,
        sim_ticks=args.ticks,
        sim_seed=args.seed,
        commodity=args.commodity,
    )

    # Initialise LLM strategist (uses real disk reads via ToolRegistry)
    strategist = AdversarialStrategist(
        config=strategist_config,
        run_dir=master_run_dir,
    )

    # ----------------------------------------------------------------
    # Main Loop
    # ----------------------------------------------------------------
    for round_num in range(args.rounds):
        # --- Step 1: LLM Mutation (skip for Round 0 — it's the baseline) ---
        if round_num > 0:
            personas_to_mutate = (
                detector_config.get("personas", ["spoofing", "wash_trading", "pump_and_dump"]) 
                if args.persona.lower() == "all" else [args.persona]
            )

            for persona in personas_to_mutate:
                logger.info(f"\nRound {round_num}: Launching Adversarial Strategist for '{persona}'...")
                try:
                    result = strategist.run_strategist_round(
                        round_num=round_num - 1,  # Strategist reads the PREVIOUS round's metrics
                        persona=persona,
                    )
                    if result.get("status") == "success":
                        logger.info(f"LLM mutation applied for {persona} in Round {round_num}!")
                    else:
                        logger.warning(f"LLM mutation failed for {persona}: {result}. Using existing config.")
                except Exception as e:
                    logger.error(f"Strategist failed for {persona}: {e}. Continuing with current config.")

        # --- Step 2: Run the simulation + detector for this round ---
        logger.info(f"\nRound {round_num}: Running simulation + detection...")
        metrics = orchestrator.run_round(round_num)

        if not metrics:
            logger.warning(f"Round {round_num} produced no metrics. Check simulation output.")
            continue

        # --- Step 3: Print running summary ---
        summary_df = build_summary_table(orchestrator._all_metrics)
        print_metrics_table(summary_df)

    # ----------------------------------------------------------------
    # Final output
    # ----------------------------------------------------------------
    final_parquet = master_run_dir / "round_metrics.parquet"
    logger.info("\n" + "=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info(f"Run directory: {master_run_dir}")
    logger.info(f"Cumulative metrics: {final_parquet}")
    logger.info("=" * 60)

    # Print final table
    summary_df = build_summary_table(orchestrator._all_metrics)
    print("\n=== FINAL ADVERSARIAL LOOP RESULTS ===")
    print_metrics_table(summary_df)


if __name__ == "__main__":
    main()

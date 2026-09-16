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
import os
from datetime import datetime, timezone
from pathlib import Path
import yaml

# Suppress pandera warnings BEFORE importing any ML/data modules
os.environ["DISABLE_PANDERA_IMPORT_WARNING"] = "True"
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="pandera.*")

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.utils.logging_config import setup_logging
setup_logging()
logger = logging.getLogger(__name__)

from src.adversarial.orchestrator import RoundOrchestrator
from src.adversarial.metrics import build_summary_table
from src.agentic.adversarial_strategist import AdversarialStrategist


def print_metrics_table(summary_df, current_round: int = -1):
    """Prints a clean per-round, per-persona metrics table to the console."""
    if summary_df.empty:
        return

    if current_round >= 0:
        print(f"\n" + "=" * 72)
        print(f"📊 LIVE PIPELINE STATUS: ROUND {current_round}".center(72))
        print("=" * 72)

        baseline_df = summary_df[summary_df['round'] == 0]
        if not baseline_df.empty:
            print("\n[BASELINE METRICS - ROUND 0 (No LLM Mutations)]")
            print(f"{'Persona':<20} {'Precision':>10} {'Recall':>8} {'F1':>8} {'FPR':>8}")
            print("-" * 60)
            for _, row in baseline_df.iterrows():
                print(f"{row['persona']:<20} {row['precision']:>10.3f} {row['recall']:>8.3f} {row['f1']:>8.3f} {row['fpr']:>8.3f}")

        if current_round > 0:
            current_df = summary_df[summary_df['round'] == current_round]
            if not current_df.empty:
                print(f"\n[MUTATED METRICS - ROUND {current_round} (Post-LLM Mutations)]")
                print(f"{'Persona':<20} {'Precision':>10} {'Recall':>8} {'F1':>8} {'FPR':>8}")
                print("-" * 60)
                for _, row in current_df.iterrows():
                    print(f"{row['persona']:<20} {row['precision']:>10.3f} {row['recall']:>8.3f} {row['f1']:>8.3f} {row['fpr']:>8.3f}")

        print("\n" + "=" * 72 + "\n")
    else:
        # Final full table print
        print("\n" + "=" * 72)
        print(f"{'Round':<8} {'Persona':<20} {'Precision':>10} {'Recall':>8} {'F1':>8} {'FPR':>8}")
        print("-" * 72)

        for _, row in summary_df.iterrows():
            print(
                f"{int(row['round']):<8} {row['persona']:<20} "
                f"{row['precision']:>10.3f} {row['recall']:>8.3f} "
                f"{row['f1']:>8.3f} {row['fpr']:>8.3f}"
            )

        print("=" * 72 + "\n")

def print_parameter_dashboard(current_round: int):
    """Reads persona_config.yaml and prints a historic parameter matrix for each persona."""
    persona_config_path = REPO_ROOT / "configs" / "persona_config.yaml"
    if not persona_config_path.exists():
        return
        
    with open(persona_config_path) as f:
        p_cfg = yaml.safe_load(f) or {}
        
    print("\n" + "*" * 72)
    print(f"🛠️  PARAMETER EVOLUTION: UP TO ROUND {current_round}".center(72))
    print("*" * 72)
    
    personas = ["spoofing", "wash_trading", "pump_and_dump"]
    
    for persona in personas:
        # Collect all parameters ever used for this persona up to current_round
        all_params = set()
        for r in range(current_round + 1):
            r_key = f"round_{r}"
            # Fallback for LLM sometimes writing to top-level if it messes up structure
            cfg_node = p_cfg.get(r_key, p_cfg)
            if persona in cfg_node:
                for param in cfg_node[persona].keys():
                    if param not in ["enabled", "start_tick", "end_tick"]:
                        all_params.add(param)
        
        if not all_params:
            continue
            
        all_params = sorted(list(all_params))
        
        print(f"\n[{persona.upper()}]")
        
        # Header
        header = f"{'Parameter':<28}"
        for r in range(current_round + 1):
            header += f" | R{r:<6}"
        print(header)
        print("-" * len(header))
        
        # Rows
        for param in all_params:
            row_str = f"{param:<28}"
            for r in range(current_round + 1):
                r_key = f"round_{r}"
                cfg_node = p_cfg.get(r_key, p_cfg)
                val = "-"
                if persona in cfg_node:
                    val = cfg_node[persona].get(param, "-")
                row_str += f" | {str(val):<7}"
            print(row_str)
            
    print("\n" + "*" * 72 + "\n")



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
        
        # --- Step 0: Print Current Parameters ---
        print_parameter_dashboard(round_num)
        
        # --- Step 1: LLM Mutation (skip for Round 0 — it's the baseline) ---
        if round_num > 0:
            personas_to_mutate = (
                detector_config.get("personas", ["spoofing", "wash_trading", "pump_and_dump"]) 
                if args.persona.lower() == "all" else [args.persona]
            )

            logger.info(f"\nRound {round_num}: Launching Adversarial Strategist for personas: {personas_to_mutate}...")
            try:
                result = strategist.run_strategist_round(
                    round_num=round_num - 1,  # Strategist reads the PREVIOUS round's metrics
                    personas=personas_to_mutate,
                )
                if result.get("status") == "success":
                    logger.info(f"LLM mutation applied for {personas_to_mutate} in Round {round_num}!")
                elif result.get("status") == "partial_success":
                    logger.warning(f"LLM mutation partially applied for: {result.get('mutated')}. Using existing config for the rest.")
                else:
                    logger.warning(f"LLM mutation failed: {result}. Using existing config.")
            except Exception as e:
                logger.error(f"Strategist failed: {e}. Continuing with current config.")
                
            # --- Save History for Dashboard ---
            persona_config_path = REPO_ROOT / "configs" / "persona_config.yaml"
            if persona_config_path.exists():
                with open(persona_config_path, "r") as f:
                    p_cfg_hist = yaml.safe_load(f) or {}
                
                # Copy current top-level mutated parameters into round_{round_num} for historic tracking
                p_cfg_hist[f"round_{round_num}"] = {}
                for p_name in ["spoofing", "wash_trading", "pump_and_dump"]:
                    if p_name in p_cfg_hist:
                        p_cfg_hist[f"round_{round_num}"][p_name] = p_cfg_hist[p_name].copy()
                        
                with open(persona_config_path, "w") as f:
                    yaml.dump(p_cfg_hist, f, default_flow_style=False)
                    
            # Print the NEWly mutated parameters
            print_parameter_dashboard(round_num)

        # --- Step 2: Run the simulation + detector for this round ---
        logger.info(f"\nRound {round_num}: Running simulation + detection...")
        metrics = orchestrator.run_round(round_num)

        if not metrics:
            logger.warning(f"Round {round_num} produced no metrics. Check simulation output.")
            continue

        # --- Step 3: Print running summary ---
        summary_df = build_summary_table(orchestrator._all_metrics)
        print_metrics_table(summary_df, round_num)

    # ----------------------------------------------------------------
    # Final output
    # ----------------------------------------------------------------
    final_parquet = master_run_dir / "round_metrics.parquet"
    logger.info("\n" + "=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info(f"Run directory: {master_run_dir}")
    logger.info(f"Cumulative metrics: {final_parquet}")
    
    # --- Auto-Visualize ---
    try:
        from scripts.visualize_results import plot_metrics
        plot_path = plot_metrics(final_parquet)
        if plot_path:
            logger.info(f"📈 Performance graphs saved to: {plot_path}")
    except Exception as e:
        logger.error(f"Failed to generate graphs: {e}")
        
    logger.info("=" * 60)

    # Print final table
    summary_df = build_summary_table(orchestrator._all_metrics)
    print("\n=== FINAL ADVERSARIAL LOOP RESULTS ===")
    print_metrics_table(summary_df)


if __name__ == "__main__":
    main()

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

# LLM output can contain non-ASCII punctuation; never let console encoding crash a round.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(errors="replace")
    except Exception:
        pass

from src.adversarial.orchestrator import RoundOrchestrator
from src.adversarial.metrics import build_summary_table
from src.agentic.adversarial_strategist import AdversarialStrategist


def _fmt(v) -> str:
    return "   n/a" if v is None or v != v else f"{v:6.3f}"


def _print_rows(df) -> None:
    print(f"{'Persona':<16} {'Phase':<6} {'Precision':>9} {'Recall':>7} {'F1':>7} {'FPR':>7} {'Events':>8} {'Latency':>8} {'Effect x':>8} {'Score':>6}")
    print("-" * 96)
    for _, row in df.sort_values(["persona", "phase"], ascending=[True, False]).iterrows():
        lat = row.get("detection_latency_ticks")
        lat_s = "  -" if lat is None or lat != lat else f"{lat:.0f}"
        print(f"{row['persona']:<16} {row['phase']:<6} {_fmt(row['precision']):>9} {_fmt(row['recall']):>7} "
              f"{_fmt(row['f1']):>7} {_fmt(row['fpr']):>7} {int(row['events_detected'])}/{int(row['events_total']):<6} {lat_s:>8} "
              f"{_fmt(row.get('effectiveness_ratio')):>8} {_fmt(row.get('mutation_score')) if row['phase'] == 'pre' else '':>6}")


def write_strategist_record(run_dir: Path, round_num: int, personas: list[str],
                            source: dict[str, str], reasoning: dict[str, str],
                            recovered: dict[str, int] | None = None,
                            guard_reverts: dict[str, dict] | None = None,
                            probe: dict[str, dict] | None = None) -> None:
    """
    round_N/strategist.json: who chose each persona's parameters for round N, the
    exact changes (old -> new, versus the parameters used in round N-1) and the
    reasoning. Structured so the dashboard never has to parse free text.
    """
    from src.core.persona_params import effective_params, resolve_persona_config
    prev_path = run_dir / f"round_{round_num - 1}" / "metrics.json"
    prev = json.loads(prev_path.read_text()).get("params_used", {}) if prev_path.exists() else {}
    cfg_path = REPO_ROOT / "configs" / "persona_config.yaml"
    active = resolve_persona_config(yaml.safe_load(cfg_path.read_text()) or {}) if cfg_path.exists() else {}
    record = {"round": round_num, "personas": {}}
    for p in personas:
        new = effective_params(active, p)
        old = prev.get(p, {})
        changes = {k: [old.get(k), v] for k, v in new.items() if old.get(k) != v}
        record["personas"][p] = {
            "source": source.get(p, "unchanged"),
            "changes": changes,
            "params": new,
            "reasoning": reasoning.get(p, ""),
        }
        if (recovered or {}).get(p) is not None:
            # effect fell below the floor last round: parameters were reset to this round's set first
            record["personas"][p]["effect_recovery_from_round"] = recovered[p]
        if (guard_reverts or {}).get(p):
            # effect was at risk: these knobs were reverted by the guard after the proposal
            record["personas"][p]["effect_guard_reverts"] = guard_reverts[p]
        if (probe or {}).get(p):
            # every candidate scored against the frozen detector on probe markets; `chosen` was kept
            record["personas"][p]["probe"] = probe[p]
    out = run_dir / f"round_{round_num}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "strategist.json").write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")


def print_metrics_table(summary_df, current_round: int = -1):
    """Per-round, per-persona metrics. 'pre' = frozen previous detector, 'post' = retrained."""
    if summary_df.empty:
        return
    rounds = [current_round] if current_round >= 0 else sorted(summary_df["round"].unique())
    for r in rounds:
        print("\n" + "=" * 76)
        print(f"ROUND {r}  (pre = detector frozen before this round's mutation, post = retrained)".center(76))
        print("=" * 76)
        _print_rows(summary_df[summary_df["round"] == r])
    print()


def print_parameter_dashboard(current_round: int):
    """Reads persona_config.yaml and prints a historic parameter matrix for each persona."""
    persona_config_path = REPO_ROOT / "configs" / "persona_config.yaml"
    if not persona_config_path.exists():
        return
        
    with open(persona_config_path) as f:
        p_cfg = yaml.safe_load(f) or {}
        
    print("\n" + "*" * 72)
    print(f"[*] PARAMETER EVOLUTION: UP TO ROUND {current_round}".center(72))
    print("*" * 72)
    
    personas = ["spoofing", "wash_trading", "pump_and_dump", "layering"]
    
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
    parser.add_argument("--commodity", type=str, default="all",
                        help="'all' (default: all 8 commodities per market) or one commodity, e.g. crude_oil_wti "
                             "(1/8 of the episodes per round; run seed314 got 4 per persona and an undertrained detector)")
    parser.add_argument("--persona",   type=str, default="all",           help="Persona to target for adversarial mutation (or 'all' for all enabled personas)")
    parser.add_argument("--mutations", choices=["llm", "hand"], default="llm",
                        help="llm = Adversarial Strategist (falls back to the hand-authored library on failure); "
                             "hand = hand-authored library only (no API calls; baseline for comparisons)")
    parser.add_argument("--test-markets", type=int, default=2, help="Held-out test markets per round")
    parser.add_argument("--run-name", type=str, default="", help="Optional label appended to the run id")
    parser.add_argument("--no-probe", action="store_true",
                        help="Do not probe-test the strategist's alternative candidates (faster, single blind proposal)")
    args = parser.parse_args()

    # ----------------------------------------------------------------
    # Setup
    # ----------------------------------------------------------------
    ts = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"adversarial_{ts}_seed{args.seed}" + (f"_{args.run_name}" if args.run_name else "")
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
    original_persona_config = persona_config_path.read_text(encoding="utf-8") if persona_config_path.exists() else None
    if persona_config_path.exists():
        with open(persona_config_path) as f:
            p_cfg = yaml.safe_load(f) or {}
        if set(p_cfg) - {"round_0"}:
            # Keep only round_0 baseline (a previous run was interrupted mid-way)
            baseline = {"round_0": p_cfg.get("round_0", {})}
            with open(persona_config_path, "w") as f:
                yaml.dump(baseline, f, default_flow_style=False)
            original_persona_config = persona_config_path.read_text(encoding="utf-8")
            logger.info("Reset persona_config.yaml to baseline for Round 0.")

    # Initialise orchestrator (runs the simulation + detector each round)
    orchestrator = RoundOrchestrator(

        master_run_dir=master_run_dir,
        detector_config=detector_config,
        sim_ticks=args.ticks,
        sim_seed=args.seed,
        commodity=args.commodity,
        n_test_markets=args.test_markets,
    )
    orchestrator.last_round = args.rounds - 1   # the final round always runs the replay check

    # Initialise LLM strategist (uses real disk reads via ToolRegistry)
    strategist = AdversarialStrategist(
        config=strategist_config,
        run_dir=master_run_dir,
    )

    # ----------------------------------------------------------------
    # Main Loop
    # ----------------------------------------------------------------
    try:
        for round_num in range(args.rounds):
        
            # --- Step 0: Print Baseline Parameters for Round 0 ---
            if round_num == 0:
                print_parameter_dashboard(round_num)
        
            # --- Step 1: LLM Mutation (skip for Round 0 — it's the baseline) ---
            if round_num > 0:
                personas_to_mutate = (
                    detector_config.get("personas", ["spoofing", "wash_trading", "pump_and_dump"]) 
                    if args.persona.lower() == "all" else [args.persona]
                )

                logger.info(f"\nRound {round_num}: mutating {personas_to_mutate} ({args.mutations})...")
                try:
                    if args.mutations == "hand":
                        from src.agentic.mutation_library import get_fallback_mutation
                        result = {"status": "success", "source": {}, "reasoning": {}}
                        result["recovered"] = {}
                        for p in personas_to_mutate:
                            st = strategist.tools.recover_effect(p)
                            if st:
                                result["recovered"][p] = st.get("anchor_round")
                            status = strategist.tools.effect_status(p)
                            fb = get_fallback_mutation(p, round_num)
                            res = strategist.tools.propose_mutation(p, fb["parameters"], fb["reasoning"])
                            if "error" not in res:
                                result["source"][p] = "hand_authored"
                                result["reasoning"][p] = fb["reasoning"]
                            if not st and (rev := strategist.tools.enforce_effect_guard(p, status)):
                                result.setdefault("guard_reverts", {})[p] = rev
                                result["reasoning"][p] = (result["reasoning"].get(p, "") + f"\n[effect guard] reverted {rev}").strip()
                    else:
                        # Candidates are scored against the detector trained at the end of the previous
                        # round, i.e. exactly the detector that will be frozen for this round. probe()
                        # anchors effects on the last COMPLETED round (round_num - 1): passing this
                        # round's number read a metrics.json that does not exist yet, so the anchored
                        # estimate was never used.
                        strategist.probe_fn = None if args.no_probe else (
                            lambda cfgs, _r=round_num - 1: orchestrator.probe(cfgs, round_num=_r))
                        result = strategist.run_strategist_round(
                            round_num=round_num - 1,  # Strategist reads the PREVIOUS round's metrics
                            personas=personas_to_mutate,
                            target_round=round_num,   # ...and mutates for THIS round
                        )
                    mutation_source = {p: result.get("source", {}).get(p, "unchanged") for p in personas_to_mutate}

                    # --- Structured record of this round's mutation (read by the dashboard) ---
                    write_strategist_record(master_run_dir, round_num, personas_to_mutate, mutation_source,
                                            result.get("reasoning", {}), result.get("recovered", {}),
                                            result.get("guard_reverts", {}), result.get("probe", {}))

                    # --- Print and Log Reasoning ---
                    reasonings = result.get("reasoning", {})
                    if reasonings:
                        reasoning_log_path = master_run_dir / "llm_reasoning.log"
                        with open(reasoning_log_path, "a", encoding="utf-8") as rf:
                            rf.write(f"\n{'='*60}\nROUND {round_num} STRATEGIST REASONING\n{'='*60}\n")
                            for persona, anchor in (result.get("recovered") or {}).items():
                                rf.write(f"\n[{persona.upper()}] effect below floor -> parameters reset to round {anchor} before mutating\n")
                            for persona, rationale in reasonings.items():
                                formatted_reasoning = f"\n[{persona.upper()}]\n{rationale}\n"
                                print(formatted_reasoning)
                                rf.write(formatted_reasoning)
                            
                    if result.get("status") == "success":
                        logger.info(f"LLM mutation applied for {personas_to_mutate} in Round {round_num}!")
                    elif result.get("status") == "partial_success":
                        logger.warning(f"LLM mutation partially applied for: {result.get('mutated')}. Using existing config for the rest.")
                    else:
                        logger.warning(f"LLM mutation failed: {result}. Using existing config.")
                except Exception as e:
                    logger.error(f"Strategist failed: {e}. Continuing with current config.")
                    mutation_source = {p: "unchanged" for p in personas_to_mutate}
                
                # --- Save History for Dashboard ---
                persona_config_path = REPO_ROOT / "configs" / "persona_config.yaml"
                if persona_config_path.exists():
                    with open(persona_config_path, "r") as f:
                        p_cfg_hist = yaml.safe_load(f) or {}
                
                    # Copy current top-level mutated parameters into round_{round_num} for historic tracking
                    p_cfg_hist[f"round_{round_num}"] = {}
                    for p_name in personas_to_mutate:
                        if p_name in p_cfg_hist:
                            p_cfg_hist[f"round_{round_num}"][p_name] = p_cfg_hist[p_name].copy()
                        
                    with open(persona_config_path, "w") as f:
                        yaml.dump(p_cfg_hist, f, default_flow_style=False)
                    
                # Print the NEWly mutated parameters
                print_parameter_dashboard(round_num)

            # --- Step 2: Run the simulation + detector for this round ---
            logger.info(f"\nRound {round_num}: Running simulation + detection...")
            metrics = orchestrator.run_round(
                round_num, mutation_source=None if round_num == 0 else mutation_source)

            if not metrics:
                logger.warning(f"Round {round_num} produced no metrics. Check simulation output.")
                continue

            # --- Step 3: Print running summary ---
            summary_df = build_summary_table(orchestrator._all_metrics)
            print_metrics_table(summary_df, round_num)

    finally:
        # Restore the baseline config (also when a run is interrupted): this run's
        # mutations live in its metrics.json files.
        if original_persona_config is not None:
            persona_config_path.write_text(original_persona_config, encoding="utf-8")
            logger.info("Restored configs/persona_config.yaml to its baseline.")


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
            logger.info(f"[+] Performance graphs saved to: {plot_path}")
    except Exception as e:
        logger.error(f"Failed to generate graphs: {e}")
        
    logger.info("=" * 60)

    # Print final table
    summary_df = build_summary_table(orchestrator._all_metrics)
    print("\n=== FINAL ADVERSARIAL LOOP RESULTS ===")
    print_metrics_table(summary_df)


if __name__ == "__main__":
    main()

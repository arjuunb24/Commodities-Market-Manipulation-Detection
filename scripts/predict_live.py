"""
scripts/predict_live.py
=======================
Score a saved real-market dataset (from scripts/fetch_market_data.py) with the
models saved by an adversarial round, then check every flagged window against
the four manipulation mechanisms. Fetching is a separate script, so one dataset
can be re-scored with different rounds or runs.

Real bars carry no account ids, so this flags WINDOWS per commodity ("minutes
that look like a persona's footprint"), not traders. Output goes to
data/live/<dataset>/predictions/<run>_r<N>_<timestamp>/ (report.md, cases.json,
minute_scores.parquet).

Usage:
    python scripts/predict_live.py                                   # latest dataset, latest saved models
    python scripts/predict_live.py --data yf_1m_20261016T1500Z_20261016T2100Z --run latest --round 3
    python scripts/predict_live.py --llm --llm-max 5                 # add an LLM second opinion (uses budget)
    python scripts/predict_live.py --personas spoofing,pump_and_dump # only the personas bars reveal best
    python scripts/predict_live.py --list                            # saved datasets and model rounds
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.live.datasets import list_datasets, resolve_dataset  # noqa: E402
from src.live.predict import run_prediction  # noqa: E402
from src.live.registry import RUNS_DIR, models_dir, saved_rounds  # noqa: E402

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
if hasattr(sys.stdout, "reconfigure"):          # Windows consoles default to cp1252
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def _list() -> None:
    print("Saved datasets (data/live/):")
    for p in list_datasets():
        print("  ", p.name)
    print("\nRuns with saved models:")
    for r in sorted(RUNS_DIR.glob("adversarial_*")):
        rounds = saved_rounds(r)
        if rounds:
            print(f"   {r.name}: rounds {rounds}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="latest", help="dataset name or folder (default: most recent fetch)")
    ap.add_argument("--run", default="latest", help="adversarial run with saved models (default: newest)")
    ap.add_argument("--round", default="latest", help="round whose models to use (default: last saved)")
    ap.add_argument("--top", type=int, default=20, help="how many flagged windows to check and report")
    ap.add_argument("--personas", default="all",
                    help="'all' or a comma list, e.g. spoofing,pump_and_dump (the two bars reveal best)")
    ap.add_argument("--llm", action="store_true", help="add an LLM review of the top windows (spends LLM budget)")
    ap.add_argument("--llm-max", type=int, default=5, help="max windows sent to the LLM")
    ap.add_argument("--list", action="store_true", help="list datasets and saved model rounds, then exit")
    args = ap.parse_args()

    if args.list:
        _list()
        return
    try:
        data = resolve_dataset(args.data)
        mdir = models_dir(args.run, args.round)
    except FileNotFoundError as exc:
        sys.exit(str(exc))

    print(f"Dataset: {data.name}")
    print(f"Models:  {mdir.relative_to(REPO_ROOT)}")
    personas = None if args.personas == "all" else [p.strip() for p in args.personas.split(",")]
    try:
        res = run_prediction(data, mdir, top=args.top, use_llm=args.llm, llm_max=args.llm_max, personas=personas)
    except ValueError as exc:
        sys.exit(str(exc))
    s, cases = res["summary"], res["cases"]
    sk = s["model_skill_on_simulator"]
    print(f"\nModel skill on held-out SIMULATED markets: mean AUC {sk['mean_auc']} ({sk['level']}); "
          f"0.5 is a coin flip.")
    print(f"Scored {s['minutes_scored']:,} minutes; flagged {s['minutes_flagged']:,} ({s['flag_rate']:.1%}) "
          f"in {s['n_cases']} windows.")
    if s["llm"]["error"]:
        print(f"LLM review skipped: {s['llm']['error']}")
    if cases:
        print(f"\nTop {min(len(cases), args.top)} windows (checker verdict on the model's persona; best mechanism match):")
        for i, c in enumerate(cases[: args.top], 1):
            ch = c.get("check", {})
            best = ch.get("best_match")
            bv = ch.get("personas", {}).get(best, {}).get("verdict") if best else "-"
            benign = "; ".join(ch.get("benign_explanations", []))
            print(f"{i:>3}. {c['commodity']:<14} {c.get('start_utc', '')} +{c['minutes']:>2}m  model: {c['model_persona']:<13} "
                  f"p={c['model_probability']:.2f}  checker: {c.get('checker_verdict_for_model_persona')}  "
                  f"| best: {best} ({bv})" + (f"\n       ordinary explanation: {benign}" if benign else ""))
            if c.get("llm_review", {}).get("verdict"):
                print(f"       LLM: {c['llm_review']['verdict']} ({c['llm_review'].get('confidence')})")
    print(f"\nVerdicts: {s['verdict_counts']}")
    print(f"Report: {Path(res['out_dir']) / 'report.md'}")


if __name__ == "__main__":
    main()

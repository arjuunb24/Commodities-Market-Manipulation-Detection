"""
scripts/replay_cases.py
=======================
Replay documented real manipulation cases (configs/case_replays.yaml) inside
simulated markets and score them with the detectors an adversarial run saved,
round by round. No LLM calls; the replay markets are simulated once and cached
in data/replays/markets/.

Usage:
    python scripts/replay_cases.py                              # newest run, all saved rounds, 3 seeds
    python scripts/replay_cases.py --run adversarial_20261018T065127Z_seed7 --rounds 0,5
    python scripts/replay_cases.py --cases sarao_layering_algorithm,coscia_panther --seeds 5
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.live.registry import resolve_run, saved_rounds  # noqa: E402
from src.replay.evaluate import DEFAULT_TICKS, load_cases, run_replays, summarise, write_report  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
for noisy in ("src.core", "src.detector"):
    logging.getLogger(noisy).setLevel(logging.WARNING)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="latest", help="adversarial run with saved models (default: newest)")
    ap.add_argument("--rounds", default="all", help="'all' or a comma list of saved rounds, e.g. 0,5")
    ap.add_argument("--cases", default="all", help="'all' or a comma list of case ids from configs/case_replays.yaml")
    ap.add_argument("--seeds", type=int, default=3, help="simulated markets per case and commodity")
    ap.add_argument("--ticks", type=int, default=DEFAULT_TICKS, help="length of each replay market (1 tick = 1 s)")
    args = ap.parse_args()

    try:
        run_dir = resolve_run(args.run)
    except FileNotFoundError as exc:
        sys.exit(str(exc))
    have = saved_rounds(run_dir)
    if not have:
        sys.exit(f"{run_dir.name} has no saved models. Backfill: python scripts/save_round_models.py --run {run_dir.name}")
    rounds = have if args.rounds == "all" else [int(r) for r in args.rounds.split(",")]
    missing = [r for r in rounds if r not in have]
    if missing:
        sys.exit(f"Rounds {missing} have no saved models (saved: {have}).")
    cases = load_cases()
    if args.cases != "all":
        wanted = [c.strip() for c in args.cases.split(",")]
        unknown = [c for c in wanted if c not in cases]
        if unknown:
            sys.exit(f"Unknown cases {unknown}. Available: {', '.join(cases)}")
        cases = {c: cases[c] for c in wanted}

    seeds = list(range(1, args.seeds + 1))
    print(f"Run {run_dir.name}, rounds {rounds}; {len(cases)} cases x their markets x {len(seeds)} seeds")
    df = run_replays(run_dir, rounds, cases, seeds, ticks=args.ticks)
    out = write_report(df, run_dir, cases)
    summ = summarise(df)
    print()
    print(summ.to_string(index=False, formatters={
        "caught": "{:.0%}".format, "recognised": "{:.0%}".format, "recognised_family": "{:.0%}".format,
        "label_share": "{:.0%}".format, "max_prob": "{:.2f}".format,
        "window_recall": "{:.0%}".format, "legit_alerts_h": "{:.1f}".format}))
    print(f"\nReport: {out / 'report.md'}")


if __name__ == "__main__":
    main()

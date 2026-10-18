"""
scripts/save_round_models.py
============================
Backfill saved models (round_N/models/) for adversarial runs made before the
orchestrator saved them itself. Nothing is simulated and no LLM is called: the
detector is retrained from the run's cached training features, round by round in
order (so threshold stickiness behaves as it did in the loop), and the bar-level
model is trained from the run's simulated trade logs.

The current configs/detector_config.yaml is used. Each round's rebuilt thresholds
are compared with the thresholds.json the run saved; the manifest records whether
they match, so you can tell a faithful rebuild from one under changed settings.

Usage:
    python scripts/save_round_models.py --run adversarial_20261016T140752Z_seed42
    python scripts/save_round_models.py --run latest --rounds 5        # one round (still trains 0..5 in order)
    python scripts/save_round_models.py --run latest --bar-only        # only the bar model (fast)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.detector.pipeline import DetectorPipeline  # noqa: E402
from src.live.bar_model import train_round_bar_model  # noqa: E402
from src.live.registry import RUNS_DIR, save_round_models  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("save_round_models")


def _run_dir(name: str) -> Path:
    if name == "latest":
        runs = sorted(RUNS_DIR.glob("adversarial_*"))
        if not runs:
            sys.exit("No adversarial runs in data/runs/.")
        return runs[-1]
    p = Path(name)
    p = p if p.is_dir() else RUNS_DIR / name
    if not p.is_dir():
        sys.exit(f"Run not found: {name}")
    return p


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="latest", help="run name or path (default: newest run)")
    ap.add_argument("--rounds", default="all", help="'all' or a comma list, e.g. 4,5")
    ap.add_argument("--bar-only", action="store_true", help="skip the trader-level detector")
    args = ap.parse_args()

    run_dir = _run_dir(args.run)
    all_rounds = sorted(int(p.name.split("_")[1]) for p in run_dir.glob("round_*") if (p / "train").is_dir())
    if not all_rounds:
        sys.exit(f"{run_dir.name} has no rounds with training data.")
    wanted = all_rounds if args.rounds == "all" else sorted(int(x) for x in args.rounds.split(","))
    missing = [r for r in wanted if r not in all_rounds]
    if missing:
        sys.exit(f"Rounds {missing} not in {run_dir.name} (has {all_rounds}).")

    config = yaml.safe_load((REPO_ROOT / "configs" / "detector_config.yaml").read_text(encoding="utf-8"))
    pipeline = DetectorPipeline(config)
    personas = config.get("personas", ["spoofing", "wash_trading", "pump_and_dump", "layering"])
    X_tr, y_tr = [], []
    logger.info("Run %s: saving models for rounds %s", run_dir.name, wanted)
    for r in range(max(wanted) + 1):
        t0 = time.time()
        rd = run_dir / f"round_{r}"
        detector = None
        extra = {"backfilled": True}
        if not args.bar_only:
            X_r, y_r = pipeline.load_and_preprocess(rd / "train")
            if not X_r.empty:
                X_tr.append(X_r)
                y_tr.append(y_r)
            # Train every round in order: thresholds are sticky across retrains, as in the loop.
            pipeline.train(pd.concat(X_tr), pd.concat(y_tr))
            detector = pipeline
            saved = rd / "thresholds.json"
            if saved.exists():
                orig = json.loads(saved.read_text(encoding="utf-8"))
                now = pipeline.supervised_detector.thresholds
                diff = {p: [round(float(orig[p]), 4), round(float(now.get(p, float("nan"))), 4)]
                        for p in orig if abs(float(orig[p]) - float(now.get(p, -1))) > 1e-6}
                extra["threshold_match"] = not diff
                if diff:
                    extra["threshold_diff_saved_vs_rebuilt"] = diff
                    logger.warning("Round %d: rebuilt thresholds differ from the run's (detector settings "
                                   "changed since the run?): %s", r, diff)
        if r not in wanted:
            continue
        bar_model = train_round_bar_model(run_dir, r, personas)
        out = save_round_models(rd, pipeline=detector, bar_model=bar_model, extra=extra)
        skill = bar_model.skill_summary()
        logger.info("Round %d saved to %s in %.0fs | bar model held-out AUC %s (%s)%s", r, out.relative_to(REPO_ROOT),
                    time.time() - t0, skill["mean_auc"], skill["level"],
                    "" if args.bar_only else f" | detector thresholds match the run: {extra.get('threshold_match')}")


if __name__ == "__main__":
    main()

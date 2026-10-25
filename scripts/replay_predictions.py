"""
scripts/replay_predictions.py
=============================
What the detector predicted on every real-case replay market, with metrics.

Scores the evaluation replay markets (configs/case_replays.yaml, seeds 1-3; the
seed-101..103 markets are replay_augmentation TRAINING data and are skipped)
with one saved round's detector and writes, under
data/replays/predictions/<run>_r<round>/:

  markets.csv    one row per market: true scheme, predicted scheme, caught /
                 right label, window-level precision / recall / F1 (any alert
                 and right label), first alert, false alerts per hour
  cases.csv      the same metrics pooled per case
  confusion.csv  episode-level confusion: true scheme x predicted scheme
  windows.csv    every alerted window and every episode window (each prediction)

Ground truth per window: one of the case's accounts AND the window overlaps
the episode by >= 15 ticks (span labels; replays carry no activity tags).
Precision counts every other alerted window as a false positive (legitimate
accounts and the replay account outside its episode); the designated market
maker (mm_*) is excluded, as everywhere in the detector.

`trained_on_case` = the case is in replay_augmentation: the detector trained on
OTHER markets of the same script, so these are seen-case, unseen-market results.

Usage:
    python scripts/replay_predictions.py                         # newest run, last saved round
    python scripts/replay_predictions.py --run adversarial_20261022T162828Z_seed222 --round 4
    python scripts/replay_predictions.py --cases sarao_layering_algorithm,coscia_panther
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.live.registry import load_detector, resolve_run, saved_rounds  # noqa: E402
from src.replay.evaluate import (  # noqa: E402
    DEFAULT_TICKS, FAKE_DEPTH_FAMILY, MIN_OVERLAP_TICKS, REPLAY_DIR, load_cases, market_dir, simulate_market,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
for noisy in ("src.core", "src.detector"):
    logging.getLogger(noisy).setLevel(logging.WARNING)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
logger = logging.getLogger("replay_predictions")


def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return round(p, 4), round(r, 4), round(2 * p * r / (p + r), 4) if p + r else 0.0


def predict_market(mdir: Path, detector) -> tuple[dict, pd.DataFrame]:
    """One replay market: the per-market summary row and its window-level predictions."""
    ev = pd.read_parquet(mdir / "manipulation_events.parquet").iloc[0]
    meta = json.loads((mdir / "replay.json").read_text(encoding="utf-8"))
    label, accounts = str(ev["persona"]), set(ev["trader_ids"])
    s, e = int(ev["start_tick"]), int(ev["end_tick"])

    X, _ = detector.load_and_preprocess(mdir)
    res = detector.score(X)
    w = res.index.to_frame(index=False)
    w["window_end"] = res["window_end"].to_numpy()
    w["final_flag"] = res["final_flag"].to_numpy()
    w["final_score"] = res["final_score"].round(4).to_numpy()
    w["predicted_persona"] = res["predicted_persona"].astype(str).to_numpy()
    w["alert_tier"] = res["alert_tier"].to_numpy() if "alert_tier" in res else ""
    w["prob_true_label"] = res[f"prob_{label}"].round(4).to_numpy() if f"prob_{label}" in res else np.nan
    w = w[~w["trader_id"].astype(str).str.startswith("mm_")].reset_index(drop=True)

    is_case = w["trader_id"].isin(accounts)
    in_episode = is_case & ((np.minimum(w["window_end"], e) - np.maximum(w["window_start"], s)) >= MIN_OVERLAP_TICKS)
    flag = w["final_flag"].astype(bool)
    right = w["predicted_persona"] == label
    family = FAKE_DEPTH_FAMILY if label in FAKE_DEPTH_FAMILY else {label}
    w["truth"] = np.where(in_episode, label, "none")
    w["outcome"] = np.select(
        [in_episode & flag & right, in_episode & flag, in_episode, flag & is_case, flag],
        ["TP right label", "TP wrong label", "FN missed", "FP case account outside episode", "FP legitimate account"],
        "TN")

    tp, fn, fp = int((in_episode & flag).sum()), int((in_episode & ~flag).sum()), int((~in_episode & flag).sum())
    tp_lbl = int((in_episode & flag & right).sum())
    p, r, f1 = _prf(tp, fp, fn)
    pl, rl, f1l = _prf(tp_lbl, fp + tp - tp_lbl, fn + tp - tp_lbl)
    hit_labels = w.loc[in_episode & flag, "predicted_persona"].value_counts()
    predicted = str(hit_labels.index[0]) if len(hit_labels) else "missed"
    hours = max((w["window_end"].max() - w["window_start"].min()) / 3600.0, 1e-9)
    first = w.loc[in_episode & flag, "window_end"].min()

    row = {
        "case": meta["case"], "commodity": meta["commodity"], "seed": meta["seed"],
        "true_label": label, "predicted_label": predicted,
        "caught": tp > 0, "label_correct": predicted == label, "family_correct": predicted in family,
        "episode_windows": tp + fn, "tp": tp, "fp": fp, "fn": fn, "tp_right_label": tp_lbl,
        "precision": p, "recall": r, "f1": f1,
        "precision_right_label": pl, "recall_right_label": rl, "f1_right_label": f1l,
        "first_alert_s": int(first - s) if tp else None,
        "max_prob_true_label": round(float(w.loc[in_episode, "prob_true_label"].max()), 4) if in_episode.any() else None,
        "false_alerts_per_h": round(int((flag & ~is_case).sum()) / hours, 2),
        "predicted_mix": json.dumps({str(k): int(v) for k, v in hit_labels.items()}),
    }
    keep = w[flag | in_episode].assign(case=meta["case"], seed=meta["seed"])
    return row, keep


def pooled(df: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """Window metrics from pooled counts, episode rates as means."""
    g = df.groupby(by, sort=False)
    out = g.agg(markets=("case", "size"), true_label=("true_label", "first"),
                caught=("caught", "mean"), label_correct=("label_correct", "mean"),
                family_correct=("family_correct", "mean"), tp=("tp", "sum"), fp=("fp", "sum"), fn=("fn", "sum"),
                tp_right_label=("tp_right_label", "sum"), first_alert_s=("first_alert_s", "median"),
                false_alerts_per_h=("false_alerts_per_h", "mean"))
    prf = [_prf(a, b, c) for a, b, c in zip(out["tp"], out["fp"], out["fn"])]
    prf_l = [_prf(t, f + a - t, n + a - t) for t, f, a, n in zip(out["tp_right_label"], out["fp"], out["tp"], out["fn"])]
    out["precision"], out["recall"], out["f1"] = zip(*prf)
    out["precision_right_label"], out["recall_right_label"], out["f1_right_label"] = zip(*prf_l)
    for c in ("caught", "label_correct", "family_correct", "false_alerts_per_h"):
        out[c] = out[c].round(3)
    return out.reset_index()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="latest", help="adversarial run with saved models (default: newest)")
    ap.add_argument("--round", default="last", help="saved round whose detector predicts (default: last)")
    ap.add_argument("--cases", default="all", help="'all' or a comma list of case ids")
    ap.add_argument("--seeds", default="1,2,3", help="evaluation replay seeds (101-103 are training markets)")
    ap.add_argument("--ticks", type=int, default=DEFAULT_TICKS)
    args = ap.parse_args()

    run_dir = resolve_run(args.run)
    have = saved_rounds(run_dir)
    if not have:
        sys.exit(f"{run_dir.name} has no saved models.")
    rnd = have[-1] if args.round == "last" else int(args.round)
    if rnd not in have:
        sys.exit(f"Round {rnd} has no saved models (saved: {have}).")
    cases = load_cases()
    if args.cases != "all":
        cases = {c.strip(): cases[c.strip()] for c in args.cases.split(",")}
    seeds = [int(s) for s in args.seeds.split(",")]
    det_cfg = yaml.safe_load((REPO_ROOT / "configs" / "detector_config.yaml").read_text(encoding="utf-8")) or {}
    trained = set((det_cfg.get("replay_augmentation") or {}).get("cases") or [])

    detector = load_detector(run_dir / f"round_{rnd}" / "models")
    rows, windows = [], []
    for cid, script in cases.items():
        for com in script.get("markets", []):
            for sd in seeds:
                mdir = market_dir(cid, com, sd, args.ticks)
                if not (mdir / "manipulation_events.parquet").exists():
                    logger.info("Simulating missing replay market %s", mdir.name)
                    simulate_market(cid, script, com, sd, args.ticks)
                row, win = predict_market(mdir, detector)
                rows.append({**row, "trained_on_case": cid in trained})
                windows.append(win)
        logger.info("Scored %s", cid)

    markets = pd.DataFrame(rows)
    by_case = pooled(markets, ["case"]).merge(markets.groupby("case")["trained_on_case"].first().reset_index(), on="case")
    markets["split"] = np.where(markets["trained_on_case"], "seen case (other markets)", "held-out case")
    overall = pooled(markets.assign(all="all"), ["all"]).drop(columns=["true_label"])
    by_split = pooled(markets, ["split"]).drop(columns=["true_label"])
    confusion = pd.crosstab(markets["true_label"], markets["predicted_label"])

    out = REPLAY_DIR / "predictions" / f"{run_dir.name}_r{rnd}"
    out.mkdir(parents=True, exist_ok=True)
    markets.to_csv(out / "markets.csv", index=False)
    by_case.to_csv(out / "cases.csv", index=False)
    confusion.to_csv(out / "confusion.csv")
    pd.concat(windows, ignore_index=True).to_csv(out / "windows.csv", index=False)

    pd.set_option("display.width", 220)
    show = ["case", "commodity", "seed", "true_label", "predicted_label", "caught", "label_correct",
            "precision", "recall", "f1", "f1_right_label", "first_alert_s", "false_alerts_per_h"]
    print(f"\nRun {run_dir.name}, round {rnd} detector, {len(markets)} replay markets\n")
    print("== Every market ==")
    print(markets[show].to_string(index=False))
    print("\n== Per case (window counts pooled) ==")
    print(by_case[["case", "trained_on_case", "markets", "true_label", "caught", "label_correct", "family_correct",
                   "precision", "recall", "f1", "f1_right_label", "first_alert_s", "false_alerts_per_h"]].to_string(index=False))
    print("\n== Held-out vs seen cases ==")
    print(by_split[["split", "markets", "caught", "label_correct", "family_correct", "precision", "recall", "f1",
                    "f1_right_label"]].to_string(index=False))
    print("\n== Overall ==")
    print(overall[["markets", "caught", "label_correct", "family_correct", "precision", "recall", "f1",
                   "precision_right_label", "recall_right_label", "f1_right_label", "false_alerts_per_h"]].to_string(index=False))
    print("\n== Episode confusion (rows: true scheme, columns: predicted) ==")
    print(confusion.to_string())
    print(f"\nWrote markets.csv, cases.csv, confusion.csv, windows.csv to {out}")


if __name__ == "__main__":
    main()

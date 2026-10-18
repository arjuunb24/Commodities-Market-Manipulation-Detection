"""
src/replay/evaluate.py
======================
Replay documented real cases in simulated markets and score them with the
detectors saved by an adversarial run.

Each replay market is an ordinary simulated market (all legitimate participants)
with the four project personas switched off and one case script running inside.
Markets are cached under data/replays/markets/, keyed by case, commodity, seed
and length, so every run and round is scored on identical markets.

Per (case, market, round) the scorer records:
  caught            any alert on the replayed account(s) in a window overlapping the episode
  recognised        at least one alerted window carries the case's own label (a persona
                    classifier fired, not only the "unusual account" IsolationForest);
                    family: spoofing and layering count for each other (both fake depth)
  label_share       share of the alerted windows with the right label (family)
  max_prob          highest classifier probability for the case's label in its windows
  window_recall     share of the account's episode windows that were alerted
  first_alert_s     seconds from episode start to the first alert
  legit_alerts_h    alerts on legitimate accounts per simulated hour (the cost side)

Comparing rounds answers whether hardening the detector against the LLM
strategist's invented evasions also helps against real, human-designed tactics.
"""

from __future__ import annotations

import json
import logging
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
CASES_PATH = REPO_ROOT / "configs" / "case_replays.yaml"
REPLAY_DIR = REPO_ROOT / "data" / "replays"
MARKETS_DIR = REPLAY_DIR / "markets"
FAKE_DEPTH_FAMILY = {"spoofing", "layering"}
MIN_OVERLAP_TICKS = 15
DEFAULT_TICKS = 6000
REPLAY_START = 2400      # leaves 40 minutes of ordinary trading before the case (feature baselines, self-history)


def load_cases(path: Path = CASES_PATH) -> dict[str, dict]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}


def market_dir(case_id: str, commodity: str, seed: int, ticks: int) -> Path:
    return MARKETS_DIR / f"{case_id}__{commodity}__s{seed}__t{ticks}"


def simulate_market(case_id: str, script: dict, commodity: str, seed: int, ticks: int = DEFAULT_TICKS,
                    start: int = REPLAY_START, force: bool = False) -> Path:
    """Simulate one replay market (cached). Returns its directory of parquet logs."""
    from src.core.calibration import CalibrationEngine
    from src.core.simulation import SimConfig, multi_commodity_seed, run_simulation
    from src.utils import io as io_utils

    out = market_dir(case_id, commodity, seed, ticks)
    if not force and (out / "manipulation_events.parquet").exists():
        return out
    out.mkdir(parents=True, exist_ok=True)
    ou, _ = CalibrationEngine().get_fallback(commodity)
    cfg = SimConfig(commodity=commodity, fundamental_value_drift=ou.drift, fundamental_value_vol=ou.volatility,
                    n_ticks=ticks, data_dir=out, overwrite=True, write_to_disk=False)
    cfg.enable_spoofing = cfg.enable_wash_trading = cfg.enable_pump_and_dump = cfg.enable_layering = False
    end = min(ticks - 300, start + int(script.get("duration_ticks", 600)))
    cfg.replay_scripts = [{"case": case_id, "script": script, "start_tick": start, "end_tick": end}]
    res = run_simulation(cfg, multi_commodity_seed(seed, commodity))
    io_utils.write_parquet(res.trade_log, out / "trade_log.parquet", "trade_log", overwrite=True)
    io_utils.write_parquet(res.order_log, out / "order_log.parquet", "order_log", overwrite=True)
    io_utils.write_parquet(res.manipulation_events, out / "manipulation_events.parquet", "manipulation_events",
                           overwrite=True)
    (out / "replay.json").write_text(json.dumps({"case": case_id, "commodity": commodity, "seed": seed, "ticks": ticks,
                                                 "start_tick": start, "end_tick": end}, indent=2), encoding="utf-8")
    return out


def _sim_job(args) -> str:
    return str(simulate_market(*args))


def score_market(mdir: Path, detector, round_num: int) -> dict[str, Any]:
    """Score one replay market with one detector."""
    mdir = Path(mdir)
    ev = pd.read_parquet(mdir / "manipulation_events.parquet")
    meta = json.loads((mdir / "replay.json").read_text(encoding="utf-8"))
    row = ev.iloc[0]
    label = str(row["persona"])
    accounts = set(row["trader_ids"])
    s, e = int(row["start_tick"]), int(row["end_tick"])
    X, _ = detector.load_and_preprocess(mdir)
    res = detector.score(X)
    idx = res.index.to_frame(index=False)
    ws = idx["window_start"].to_numpy()
    we = res["window_end"].to_numpy()
    trader = idx["trader_id"].to_numpy()
    is_rep = np.isin(trader, list(accounts))
    overlap = np.minimum(we, e) - np.maximum(ws, s) >= MIN_OVERLAP_TICKS
    rep = is_rep & overlap
    flag = res["final_flag"].to_numpy()
    hit = rep & flag
    labels = res["predicted_persona"].to_numpy()[hit]
    personas = pd.Series(labels).value_counts()
    top = str(personas.index[0]) if len(personas) else None
    family = FAKE_DEPTH_FAMILY if label in FAKE_DEPTH_FAMILY else {label}
    in_family = np.isin(labels.astype(str), list(family))
    hours = max((we.max() - ws.min()) / 3600.0, 1e-9) if len(we) else 1.0
    legit = ~is_rep & ~pd.Series(trader).str.startswith("mm_").to_numpy() & flag
    prob_col = f"prob_{label}"
    return {
        "case": meta["case"], "commodity": meta["commodity"], "seed": meta["seed"], "round": round_num,
        "label": label,
        "episode_windows": int(rep.sum()),
        "caught": bool(hit.any()),
        "alerted_as": top,
        "labels": json.dumps({str(k): int(v) for k, v in personas.items()}),
        "recognised": bool((labels.astype(str) == label).any()),
        "recognised_family": bool(in_family.any()),
        "label_share": round(float(in_family.mean()), 4) if len(labels) else 0.0,
        "window_recall": round(float(hit.sum() / max(rep.sum(), 1)), 4),
        "first_alert_s": int(we[hit].min() - s) if hit.any() else None,
        "max_prob": round(float(res.loc[rep, prob_col].max()), 4) if rep.any() and prob_col in res else None,
        "threshold": round(float(detector.supervised_detector.thresholds.get(label, float("nan"))), 4),
        "legit_alerts_h": round(float(legit.sum() / hours), 2),
    }


def heldout_reference(run_dir: Path, round_num: int) -> dict[str, float | None]:
    """The run's own held-out episode detection rate per persona (invented personas), for comparison."""
    path = Path(run_dir) / f"round_{round_num}" / "metrics.json"
    if not path.exists():
        return {}
    m = json.loads(path.read_text(encoding="utf-8")).get("personas", {})
    out = {}
    for p, v in m.items():
        tot = v.get("events_total") or 0
        out[p] = round(v.get("events_detected", 0) / tot, 4) if tot else None
    return out


def run_replays(run_dir: Path, rounds: list[int], cases: dict[str, dict], seeds: list[int],
                ticks: int = DEFAULT_TICKS, workers: int | None = None) -> pd.DataFrame:
    from src.live.registry import load_detector

    jobs = [(cid, sc, com, sd, ticks) for cid, sc in cases.items() for com in sc.get("markets", []) for sd in seeds]
    todo = [j for j in jobs if not (market_dir(j[0], j[2], j[3], j[4]) / "manipulation_events.parquet").exists()]
    if todo:
        logger.info("Simulating %d replay markets (%d cached)...", len(todo), len(jobs) - len(todo))
        with ProcessPoolExecutor(max_workers=workers) as ex:
            list(ex.map(_sim_job, todo))
    dirs = [market_dir(j[0], j[2], j[3], j[4]) for j in jobs]
    rows = []
    for r in rounds:
        det = load_detector(Path(run_dir) / f"round_{r}" / "models")
        for d in dirs:
            rows.append(score_market(d, det, r))
        logger.info("Round %d scored on %d replay markets", r, len(dirs))
    return pd.DataFrame(rows)


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["case", "round"])
    out = pd.DataFrame({
        "markets": g.size(),
        "caught": g["caught"].mean(),
        "recognised": g["recognised"].mean(),
        "recognised_family": g["recognised_family"].mean(),
        "label_share": g["label_share"].mean(),
        "max_prob": g["max_prob"].median(),
        "window_recall": g["window_recall"].mean(),
        "first_alert_s": g["first_alert_s"].median(),
        "legit_alerts_h": g["legit_alerts_h"].mean(),
    }).reset_index()
    return out


def write_report(df: pd.DataFrame, run_dir: Path, cases: dict, out_root: Path = REPLAY_DIR) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = Path(out_root) / f"{Path(run_dir).name}_{stamp}"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "results.csv", index=False)
    summ = summarise(df)
    summ.to_csv(out / "summary.csv", index=False)
    rounds = sorted(df["round"].unique())
    first, last = rounds[0], rounds[-1]
    L = [f"# Real-case replay test: {Path(run_dir).name}", "",
         "Documented real manipulation cases replayed as scripts inside simulated markets and scored with the "
         "detectors this run saved. The strategist never saw these cases and no detector trained on them.", "",
         "## Did the detector catch the real tactics?", "",
         "Caught = any alert on the account. Recognised = a persona classifier gave the right label in at least "
         "one alerted window (family: spoofing and layering count for each other). Label share = share of alerted "
         "windows with the right label; the rest are IsolationForest 'unusual account' alerts or another persona.", "",
         "| Case | Label | Round | Caught | Recognised | Right family | Label share | Max prob | Window recall | First alert (s) | Legit alerts / h |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in summ.iterrows():
        lab = cases.get(r["case"], {}).get("label", "")
        fa = "-" if pd.isna(r["first_alert_s"]) else f"{r['first_alert_s']:.0f}"
        mp = "-" if pd.isna(r["max_prob"]) else f"{r['max_prob']:.2f}"
        L.append(f"| {r['case']} | {lab} | {r['round']} | {r['caught']:.0%} | {r['recognised']:.0%} | {r['recognised_family']:.0%} | "
                 f"{r['label_share']:.0%} | {mp} | {r['window_recall']:.0%} | {fa} | {r['legit_alerts_h']:.1f} |")
    L += ["", f"## Does training against the LLM strategist generalise? (round {first} vs round {last})", ""]
    for cid in summ["case"].unique():
        a = summ[(summ["case"] == cid) & (summ["round"] == first)].iloc[0]
        b = summ[(summ["case"] == cid) & (summ["round"] == last)].iloc[0]
        L.append(f"- **{cid}**: recognised {a['recognised']:.0%} -> {b['recognised']:.0%} (family "
                 f"{a['recognised_family']:.0%} -> {b['recognised_family']:.0%}), max probability "
                 f"{a['max_prob']:.2f} -> {b['max_prob']:.2f}, window recall {a['window_recall']:.0%} -> {b['window_recall']:.0%}")
    ref = heldout_reference(run_dir, last)
    if ref:
        L += ["", f"For comparison, round {last}'s detector caught these shares of the run's own (invented) held-out episodes: "
              + ", ".join(f"{p.replace('_', ' ')} {v:.0%}" for p, v in ref.items() if v is not None) + "."]
    L += ["", "## Cases", ""]
    for cid, c in cases.items():
        L.append(f"**{cid}**: {c.get('title', '')}  ")
        L.append("Verified: " + "; ".join(c.get("verified", [])) + ".  ")
        L.append("Approximated: " + "; ".join(c.get("approximated", [])) + ".  ")
        L.append("Sources: " + "; ".join(c.get("sources", [])))
        L.append("")
    L += ["---", "Scripts are stylised from public records, not reconstructed order logs. A catch shows the detector "
          "recognises the documented pattern in a simulated market; it is not a test on the real market data."]
    (out / "report.md").write_text("\n".join(L), encoding="utf-8")
    return out

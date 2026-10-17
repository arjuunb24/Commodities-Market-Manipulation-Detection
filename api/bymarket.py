"""
api/bymarket.py
===============
Round metrics for ONE test market.

Runs made after the per-market update store them in metrics.json["by_market"]
(frozen and retrained detector, effect, scores). For older runs the retrained
metrics and the effect are derived here exactly from what the run saved per
market (flags.parquet persona scores + thresholds, labels.parquet, the market's
episode and trade/order logs); the frozen-detector numbers were never stored per
market, so they are reported as unavailable rather than approximated.
"""

from __future__ import annotations

import json
import logging
import threading

import numpy as np
import pandas as pd

from api import data
from api.data import clean

logger = logging.getLogger(__name__)

INDEX = ["commodity", "window_start", "trader_id"]
DERIVED_VERSION = 1          # bump when the derivation changes, to recompute cached by_market.json files
_disk_lock = threading.Lock()


@data.ttl_cache(900, maxsize=8)
def _flags_scored_round(run: str, rnd: int) -> pd.DataFrame:
    """Every window of the round (all markets) with flag_<persona> rebuilt from the stored scores and thresholds."""
    import pyarrow.parquet as pq
    p = data.round_dir(run, rnd) / "flags.parquet"
    if not p.exists():
        return pd.DataFrame()
    table = pq.read_table(p)
    cols = {c: table.column(c).to_numpy(zero_copy_only=False) for c in ["commodity", "window_start", "trader_id", "window_end", "final_flag"]}
    df = pd.DataFrame(cols)
    thr = data.thresholds(run, rnd)
    if "persona_scores" in table.column_names:
        scores = table.column("persona_scores").combine_chunks()
        for i in range(scores.type.num_fields):
            name = scores.type.field(i).name
            vals = scores.field(i).to_numpy(zero_copy_only=False).astype(float)
            df[f"prob_{name}"] = np.nan_to_num(vals, nan=0.0)
            df[f"flag_{name}"] = df[f"prob_{name}"] >= thr.get(name, 1.0)
    return df


def _flags_scored(run: str, rnd: int, market: str) -> pd.DataFrame:
    """One market's windows (flags.parquet covers every test market of the round; read once per round)."""
    df = _flags_scored_round(run, rnd)
    if df.empty:
        return df
    rng = data.tick_range(run, rnd, market)
    if rng is not None:
        df = df[(df["window_start"] >= rng[0]) & (df["window_start"] <= rng[1])]
    return df.set_index(INDEX)


def _episode_orders(d, ev: pd.DataFrame) -> pd.DataFrame:
    """
    Only the order-log rows the effect metrics read: orders of the spoofing / layering
    episode accounts. Filtering at read time skips most of a ~2M-row log (17 s -> ~2 s
    for a run's five rounds).
    """
    cols = ["order_id", "trader_id", "side", "price", "quantity", "tick", "event_type", "commodity"]
    accts = sorted({a for _, e in ev[ev["persona"].isin(["spoofing", "layering"])].iterrows() for a in e["trader_ids"]}) if not ev.empty else []
    if d is None or not accts or not (d / "order_log.parquet").exists():
        return pd.DataFrame(columns=cols)
    return pd.read_parquet(d / "order_log.parquet", filters=[("trader_id", "in", accts)])


def _labels(d) -> pd.DataFrame:
    y = pd.read_parquet(d / "labels.parquet")
    return y.set_index(INDEX) if set(INDEX) <= set(y.columns) else y


@data.ttl_cache(3600, maxsize=256)   # finished runs never change
def market_metrics(run: str, rnd: int, market: str) -> dict:
    m = data.metrics(run, rnd)
    saved = (m.get("by_market") or {}).get(market)
    if saved:
        return clean({**saved, "source": "saved", "has_frozen": "pre_retrain" in saved})

    # Derived once per run and kept next to it, so later views and server restarts are instant.
    disk = data.round_dir(run, rnd) / "by_market.json"
    try:
        cached = json.loads(disk.read_text(encoding="utf-8")) if disk.exists() else {}
    except (OSError, ValueError):
        cached = {}
    if market in cached and cached[market].get("version") == DERIVED_VERSION:
        return cached[market]
    out = _derive(run, rnd, market, m)
    if "personas" not in out or "effectiveness" not in out:   # never persist a partial result
        return out
    try:
        with _disk_lock:
            current = json.loads(disk.read_text(encoding="utf-8")) if disk.exists() else {}
            current[market] = out
            disk.write_text(json.dumps(current), encoding="utf-8")
    except (OSError, ValueError) as exc:   # read-only data dir: still serve the computed numbers
        logger.warning("could not cache per-market metrics in %s: %s", disk, exc)
    return out


def _derive(run: str, rnd: int, market: str, m: dict) -> dict:
    from src.adversarial.effectiveness import effectiveness_ratio, episode_effectiveness, summarize
    from src.adversarial.metrics import compute_round_metrics
    personas = data.PERSONAS
    d = next((x for x in data.test_dirs(run, rnd, market)), None)
    out: dict = {"source": "derived", "has_frozen": False, "version": DERIVED_VERSION}
    try:
        res = _flags_scored(run, rnd, market)
        y = _labels(d) if d is not None and (d / "labels.parquet").exists() else pd.DataFrame()
        ev = data.events(run, rnd, market)
        if not res.empty and not y.empty and any(f"flag_{p}" in res.columns for p in personas):
            out["personas"] = compute_round_metrics(res, y, personas, events=ev)
    except Exception as exc:   # the market view degrades to "not available", never to an error page
        logger.warning("per-market metrics failed for %s r%s %s: %s", run, rnd, market, exc)
    try:
        ev = data.events(run, rnd, market)
        # trades read directly (not via the shared cache) so warming every round does not pin all logs in memory
        per = episode_effectiveness(ev, data._read(run, rnd, "trade_log.parquet", market), _episode_orders(d, ev))
        e = summarize(per) if not per.empty else {}
        base = (data.metrics(run, 0).get("effectiveness") or {}) if rnd != 0 else (m.get("effectiveness") or {})
        out["effectiveness"] = e
        out["effectiveness_ratio"] = {p: effectiveness_ratio(e.get(p), base.get(p), p) if p in e else None for p in personas}
        out["n_episodes"] = int(len(ev))
    except Exception as exc:
        logger.warning("per-market effect failed for %s r%s %s: %s", run, rnd, market, exc)
    return clean(out)

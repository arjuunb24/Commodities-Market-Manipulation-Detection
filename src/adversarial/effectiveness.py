"""
src/adversarial/effectiveness.py
================================
Did the manipulation still WORK? Evasion only counts if the scheme keeps doing
what it is for — otherwise "evading" is just not manipulating.

Per episode, from the test market's own logs (ground-truth side of the simulator;
never used by the detector):

  spoofing       fake_depth_share   share of the visible near-touch depth (±0.2% band) on
                                    the spoofed side that the spoof made up when placed —
                                    the lever spoofing pulls; recorded by the agent
                 favorable_move_bp  price move toward the genuine order while the spoof
                                    rested (reported, not optimised: per-cycle impact in
                                    this market is small next to noise)
  layering       fake_depth_share   same, for the stacked layers
                 genuine_fill_rate  filled / placed volume of the genuine orders
  wash_trading   wash_volume_share  self-crossed volume / all market volume in the
                                    episode window (artificial volume printed)
  pump_and_dump  ignition_pressure  peak share of market volume taken by the coalition's
                                    marketable buying in any 60-tick window (size AND
                                    concentration of the push that moves the price)
                 price_lift_bp      price move across that span (reported, not optimised:
                                    in this market hedgers absorb most pumps, so realised
                                    lift per episode is noise-dominated)
                 pnl_bp             coalition mark-to-market P&L / bought notional

PRIMARY_METRIC names the number the strategist must keep up. Order-id tags
("s" spoof leg, "g" genuine leg) are simulator audit markers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

PRIMARY_METRIC = {
    "spoofing": "fake_depth_share",
    "wash_trading": "wash_volume_share",
    "pump_and_dump": "ignition_pressure",
    "layering": "fake_depth_share",
}
EFFECTIVENESS_FLOOR = 0.5   # a mutation must keep >= 50% of the round-0 effect
EFFECT_AT_RISK = 0.65       # early warning: the next effect-reducing change likely crosses the floor
EFFECT_SURPLUS = 1.5        # effect this far above round 0 is unused budget that could buy evasion
PRESSURE_WINDOW = 60        # ticks


def _price_path(trades: pd.DataFrame) -> pd.Series:
    if trades.empty:
        return pd.Series(dtype=float)
    px = trades.groupby("tick")["price"].last()
    return px.reindex(range(int(px.index.min()), int(px.index.max()) + 1)).ffill()


def _spoofing(e, trades, orders, px, tag: str = "s") -> dict:
    accts = set(e["trader_ids"])
    o = orders[orders["trader_id"].isin(accts)]
    placed = o[o["event_type"] == "placed"]
    spoof = placed[placed["order_id"].str.endswith(tag) & placed["tick"].between(e["start_tick"], e["end_tick"])]
    genuine = placed[placed["order_id"].str.endswith("g") & placed["tick"].between(e["start_tick"], e["end_tick"])]
    params = e["parameters"] if isinstance(e["parameters"], dict) else {}
    depth = float(params.get("realised_fake_depth_share") or 0.0)
    if spoof.empty or px.empty:
        return {"fake_depth_share": depth, "favorable_move_bp": 0.0, "genuine_fill_rate": 0.0, "spoof_cycles": 0}
    cancels = o[(o["event_type"] == "cancelled") & o["order_id"].isin(spoof["order_id"])]
    end_tick = cancels.groupby("order_id")["tick"].max()
    moves = []
    for t, cyc in spoof.groupby("tick"):
        side = cyc["side"].iloc[0]
        t_end = int(end_tick.reindex(cyc["order_id"]).max()) if end_tick.reindex(cyc["order_id"]).notna().any() else t + 5
        p0, p1 = px.get(t - 1, np.nan), px.get(t_end, np.nan)
        if np.isnan(p0) or np.isnan(p1) or p0 <= 0:
            continue
        sign = 1.0 if side == "buy" else -1.0   # fake bids should lift the price for the genuine sell
        moves.append(sign * (p1 - p0) / p0 * 1e4)
    fills = o[o["event_type"].isin(["filled", "partial_fill"]) & o["order_id"].isin(genuine["order_id"])]
    g_placed = float(genuine["quantity"].sum())
    return {
        "fake_depth_share": round(depth, 4),
        "favorable_move_bp": round(float(np.mean(moves)), 4) if moves else 0.0,
        "genuine_fill_rate": round(min(1.0, float(fills["quantity"].sum()) / g_placed), 4) if g_placed else 0.0,
        "spoof_cycles": int(spoof["tick"].nunique()),
    }


def _wash(e, trades) -> dict:
    accts = set(e["trader_ids"])
    w = trades[trades["tick"].between(e["start_tick"], e["end_tick"])]
    if w.empty:
        return {"wash_volume_share": 0.0, "wash_volume": 0}
    self_x = w[w["buyer_trader_id"].isin(accts) & w["seller_trader_id"].isin(accts)]["quantity"].sum()
    return {"wash_volume_share": round(float(self_x) / float(w["quantity"].sum()), 4), "wash_volume": int(self_x)}


def _pump(e, trades, px) -> dict:
    """
    price_lift_bp: price move across the coalition's own aggressive-buying span
    (the ignition), from the last price before its first marketable buy to the
    price 5 ticks after its last one. Measuring over the ignition rather than the
    whole episode keeps unrelated drift out of the number.
    """
    accts = set(e["trader_ids"])
    if px.empty:
        return {"ignition_pressure": 0.0, "price_lift_bp": 0.0, "pnl_bp": 0.0}
    start, end = int(e["start_tick"]), int(e["end_tick"])
    t = trades[trades["tick"].between(start, end)]
    ign = t[t["buyer_trader_id"].isin(accts) & (t["aggressor_side"] == "buy")]
    lift = 0.0
    pressure = 0.0
    if not ign.empty:
        t0, t1 = int(ign["tick"].min()), int(ign["tick"].max())
        ticks = range(start, end + 1)
        push = ign.groupby("tick")["quantity"].sum().reindex(ticks, fill_value=0).rolling(PRESSURE_WINDOW).sum()
        market = t.groupby("tick")["quantity"].sum().reindex(ticks, fill_value=0).rolling(PRESSURE_WINDOW).sum()
        share = (push / market.where(market > 0)).dropna()
        pressure = float(share.max()) if len(share) else 0.0
        p0 = px.get(t0 - 1, np.nan)
        p1 = px.get(min(t1 + 5, int(px.index.max())), np.nan)
        if p0 == p0 and p1 == p1 and p0 > 0:
            lift = (p1 - p0) / p0 * 1e4
    buys = t[t["buyer_trader_id"].isin(accts)]
    sells = t[t["seller_trader_id"].isin(accts)]
    bought = float((buys["price"] * buys["quantity"]).sum())
    sold = float((sells["price"] * sells["quantity"]).sum())
    inventory = float(buys["quantity"].sum() - sells["quantity"].sum())
    mark = float(px.get(min(end, int(px.index.max())), px.iloc[-1]))
    pnl = sold - bought + inventory * mark
    return {"ignition_pressure": round(pressure, 4), "price_lift_bp": round(lift, 3),
            "pnl_bp": round(pnl / bought * 1e4, 3) if bought else 0.0}


def episode_effectiveness(events: pd.DataFrame, trades: pd.DataFrame, orders: pd.DataFrame) -> pd.DataFrame:
    """One row per ground-truth episode with persona-specific effectiveness metrics."""
    rows = []
    for comm, ev in events.groupby("commodity"):
        t = trades[trades["commodity"] == comm]
        o = orders[orders["commodity"] == comm]
        px = _price_path(t)
        for _, e in ev.iterrows():
            if e["persona"] == "spoofing":
                m = _spoofing(e, t, o, px)
            elif e["persona"] == "wash_trading":
                m = _wash(e, t)
            elif e["persona"] == "pump_and_dump":
                m = _pump(e, t, px)
            elif e["persona"] == "layering":
                m = _spoofing(e, t, o, px, tag="l")   # same economics: fake depth vs. a genuine order
            else:
                continue
            rows.append({"event_id": e["event_id"], "persona": e["persona"], **m})
    return pd.DataFrame(rows)


def summarize(per_episode: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Median of every metric per persona (+ number of episodes); medians keep a few extreme episodes from swinging the round."""
    out: dict[str, dict[str, float]] = {}
    if per_episode.empty:
        return out
    for persona, g in per_episode.groupby("persona"):
        num = g.drop(columns=["event_id", "persona"]).dropna(axis=1, how="all").select_dtypes("number")
        d = {k: round(float(v), 4) for k, v in num.median().items()}
        d["episodes"] = int(len(g))
        d["primary_metric"] = PRIMARY_METRIC.get(persona)
        out[persona] = d
    return out


def effectiveness_ratio(current: dict, baseline: dict, persona: str) -> float | None:
    """current primary / round-0 primary (None when undefined)."""
    key = PRIMARY_METRIC[persona]
    c, b = (current or {}).get(key), (baseline or {}).get(key)
    if c is None or b is None or b <= 0:
        return None
    return round(c / b, 3)


DELAY_REFERENCE_TICKS = 300   # detection later than ~5 simulated minutes counts as fully delayed
EFFECT_CAP = 1.0               # effect above round 0 earns nothing extra
SCORE_WEIGHTS = {"coverage": 0.5, "missed": 0.3, "delay": 0.2}


def evasion_components(pre_metrics: dict | None) -> dict | None:
    """
    How well a mutation evaded the FROZEN detector, from its pre-retrain metrics:
      coverage  1 - episode coverage (share of each episode's windows NOT flagged;
                falls back to 1 - row recall for older metrics)
      missed    share of episodes never flagged at all
      delay     detection delay of caught episodes / DELAY_REFERENCE_TICKS (capped at 1;
                1 when every episode was missed)
    "Flagged" means ANY alert (caught_* metrics) when available, so being flagged
    under another persona's label is not evasion. Older metrics without caught_*
    fall back to the persona's own classifier.
    """
    if not pre_metrics:
        return None
    any_alert = pre_metrics.get("caught_coverage") is not None and pre_metrics.get("caught_coverage") == pre_metrics.get("caught_coverage")
    cov = pre_metrics.get("caught_coverage") if any_alert else pre_metrics.get("episode_coverage")
    if cov is None or cov != cov:
        cov = pre_metrics.get("recall")
    if cov is None or cov != cov:
        return None
    total = pre_metrics.get("events_total") or 0
    detected = pre_metrics.get("caught_events_detected" if any_alert else "events_detected") or 0
    missed = (total - detected) / total if total else 0.0
    lat = pre_metrics.get("caught_latency_ticks" if any_alert else "detection_latency_ticks")
    if lat is None or lat != lat:
        delay = 1.0 if total and detected == 0 else 0.0
    else:
        delay = min(1.0, float(lat) / DELAY_REFERENCE_TICKS)
    return {"coverage": round(1.0 - float(cov), 4), "missed": round(missed, 4), "delay": round(delay, 4)}


def mutation_score(pre_metrics: dict | None, eff_ratio: float | None) -> float | None:
    """
    Strategist objective: evasion that still works.

        evasion = 0.5 x (1 - episode coverage) + 0.3 x episodes missed + 0.2 x delay
        score   = evasion x min(1, effectiveness ratio)

    Zero when the scheme kept less than EFFECTIVENESS_FLOOR of its round-0 effect
    ("evasion" by not manipulating). Effect above round 0 earns nothing extra.
    `pre_metrics` may also be a bare frozen recall (float) for backward compatibility.
    """
    if isinstance(pre_metrics, (int, float)):
        pre_metrics = {"recall": float(pre_metrics)}
    comp = evasion_components(pre_metrics)
    if comp is None or eff_ratio is None:
        return None
    if eff_ratio < EFFECTIVENESS_FLOOR:
        return 0.0
    evasion = sum(SCORE_WEIGHTS[k] * comp[k] for k in SCORE_WEIGHTS)
    return round(evasion * min(EFFECT_CAP, eff_ratio), 4)

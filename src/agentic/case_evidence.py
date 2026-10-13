"""
src/agentic/case_evidence.py
============================
Deterministic behavioural evidence for one flagged case — WHAT the account
actually did in the flagged span, measured from its own orders and trades.

The case-note LLM (Module 4A) receives this alongside the SHAP attributions so
it can explain the manipulation MECHANISM ("placed 56 large orders, cancelled
them after ~5 ticks while its small orders on the other side filled"), not just
which features fired. Every number comes straight from the logs; the
narrative's grounding check rejects any number the model did not receive.
Values are pre-rounded so the model can quote them verbatim.

Inputs are the SELECTED test market's logs (trade_log, order_log); flags are
optional and only used to list other accounts flagged for the same persona.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

LARGE_ORDER_MULTIPLE = 4.0   # same definition as the detector's "large order"
BURST_WINDOW = 60            # ticks

# How each scheme works — plain words, no numbers (keeps the grounding check strict).
MECHANISMS = {
    "spoofing": (
        "Spoofing: the trader rests a small genuine order on one side of the book, then places large orders on the "
        "OTHER side that it does not intend to execute. The fake depth makes other participants believe there is "
        "strong buying (or selling) interest, nudging prices toward the genuine order, which then fills. The large "
        "orders are cancelled within seconds, before they can trade."
    ),
    "wash_trading": (
        "Wash trading: two or more colluding accounts trade with each other at pre-arranged prices and sizes. Volume "
        "is printed to the tape, but because the same group is on both sides, their combined position barely changes. "
        "The aim is to fake activity and liquidity, misleading others about genuine interest in the contract."
    ),
    "pump_and_dump": (
        "Pump-and-dump (momentum ignition): coordinated accounts first build a long position quietly, then fire a burst "
        "of aggressive buy orders to push the price up and trigger trend-following traders. While those followers keep "
        "buying, the coalition sells its inventory into the demand it created, and the price falls back."
    ),
    "anomaly": (
        "No specific manipulation type was identified: the unsupervised detector found this account's behaviour "
        "unusual compared with all other accounts. It may be legitimate but atypical activity."
    ),
}


def _bp(a: float, b: float) -> float:
    return round((b - a) / a * 1e4, 1) if a and a == a and b == b else 0.0


def _price_path(trades: pd.DataFrame) -> pd.Series:
    if trades.empty:
        return pd.Series(dtype=float)
    px = trades.groupby("tick")["price"].last()
    return px.reindex(range(int(px.index.min()), int(px.index.max()) + 1)).ffill()


def _account_legs(trades: pd.DataFrame, trader: str) -> pd.DataFrame:
    """One row per trade the account took part in, with its side, counterparty and aggressor flag."""
    b = trades[trades["buyer_trader_id"] == trader].assign(side="buy", cpty=lambda d: d["seller_trader_id"])
    s = trades[trades["seller_trader_id"] == trader].assign(side="sell", cpty=lambda d: d["buyer_trader_id"])
    legs = pd.concat([b, s], ignore_index=True)
    if not legs.empty:
        legs["aggressive"] = legs["aggressor_side"] == legs["side"]
    return legs.sort_values("tick") if not legs.empty else legs


def _spoofing(trader, start, end, trades, orders, px) -> dict:
    span_orders = orders[orders["tick"].between(start, end)]
    placed_all = span_orders[span_orders["event_type"] == "placed"]
    median_size = float(placed_all["quantity"].median()) if not placed_all.empty else 1.0
    acct = orders[orders["trader_id"] == trader]
    placed = acct[(acct["event_type"] == "placed") & acct["tick"].between(start, end)]
    large = placed[placed["quantity"] >= LARGE_ORDER_MULTIPLE * max(median_size, 1.0)]
    cancels = acct[(acct["event_type"] == "cancelled") & acct["order_id"].isin(large["order_id"])]
    life = cancels.groupby("order_id")["tick"].min()
    large = large.assign(cancel_tick=large["order_id"].map(life))
    cancelled = large.dropna(subset=["cancel_tick"])
    legs = _account_legs(trades[trades["tick"].between(start, end + 2)], trader)

    with_opp, opp_vol, moves = 0, 0, []
    for _, o in cancelled.iterrows():
        t0, t1 = int(o["tick"]), int(o["cancel_tick"])
        if not legs.empty:
            opp = legs[(legs["tick"] >= t0) & (legs["tick"] <= t1 + 2) & (legs["side"] != o["side"])]
            if not opp.empty:
                with_opp += 1
                opp_vol += int(opp["quantity"].sum())
        p0, p1 = px.get(t0 - 1, np.nan), px.get(t1, np.nan)
        if p0 == p0 and p1 == p1:
            sign = 1.0 if o["side"] == "buy" else -1.0   # fake bids should lift the price for the genuine sell
            moves.append(sign * (p1 - p0) / p0 * 1e4)
    placed_vol = float(placed["quantity"].sum())
    cancel_vol = float(acct[(acct["event_type"] == "cancelled") & acct["tick"].between(start, end)]["quantity"].sum())
    small = placed[~placed["order_id"].isin(large["order_id"])]
    return {
        "orders_placed": int(len(placed)),
        "large_orders_placed": int(len(large)),
        "large_orders_cancelled": int(len(cancelled)),
        "large_order_median_size": int(large["quantity"].median()) if len(large) else 0,
        "market_median_order_size": int(round(median_size)),
        "large_order_size_vs_market_median_x": round(float(large["quantity"].median()) / median_size, 1) if len(large) and median_size else 0.0,
        "median_large_order_lifetime_ticks": int((cancelled["cancel_tick"] - cancelled["tick"]).median()) if len(cancelled) else None,
        "large_order_sides": {k: int(v) for k, v in large["side"].value_counts().items()},
        "small_orders_placed": int(len(small)),
        "small_order_median_size": int(small["quantity"].median()) if len(small) else 0,
        "large_orders_with_opposite_side_fill_while_resting": int(with_opp),
        "opposite_side_volume_filled": int(opp_vol),
        "avg_price_move_toward_filled_side_bp": round(float(np.mean(moves)), 2) if moves else 0.0,
        "account_cancelled_volume_pct": round(100 * cancel_vol / placed_vol, 1) if placed_vol else 0.0,
    }


def _wash(trader, start, end, trades) -> dict:
    span = trades[trades["tick"].between(start, end)]
    legs = _account_legs(span, trader)
    if legs.empty:
        return {"trades": 0}
    gross = float(legs["quantity"].sum())
    by_cpty = legs.groupby("cpty")["quantity"].sum().sort_values(ascending=False)
    top = by_cpty.index[0]
    with_top = legs[legs["cpty"] == top]
    net = int(legs[legs["side"] == "buy"]["quantity"].sum() - legs[legs["side"] == "sell"]["quantity"].sum())
    mkt_vol = float(span["quantity"].sum())
    vwap = float((span["price"] * span["quantity"]).sum() / mkt_vol) if mkt_vol else np.nan
    dev = (legs["price"] - vwap).abs() / vwap * 1e4 if vwap == vwap else pd.Series(dtype=float)
    pair_vol = float(span[((span["buyer_trader_id"] == trader) & (span["seller_trader_id"] == top))
                          | ((span["buyer_trader_id"] == top) & (span["seller_trader_id"] == trader))]["quantity"].sum())
    return {
        "trades": int(len(legs)),
        "gross_volume": int(gross),
        "net_position_change": net,
        "net_position_vs_gross_pct": round(100 * abs(net) / gross, 1) if gross else 0.0,
        "top_counterparty": str(top),
        "volume_with_top_counterparty_pct": round(100 * float(by_cpty.iloc[0]) / gross, 1),
        "trades_with_top_counterparty": int(len(with_top)),
        "bought_from_top_counterparty": int(with_top[with_top["side"] == "buy"]["quantity"].sum()),
        "sold_to_top_counterparty": int(with_top[with_top["side"] == "sell"]["quantity"].sum()),
        "other_counterparties": {str(k): int(v) for k, v in by_cpty.iloc[1:4].items()},
        "pair_share_of_market_volume_pct": round(100 * pair_vol / mkt_vol, 1) if mkt_vol else 0.0,
        "median_trade_size": int(legs["quantity"].median()),
        "median_price_distance_from_vwap_bp": round(float(dev.median()), 2) if len(dev) else None,
        "aggressive_share_pct": round(100 * float(legs[legs["aggressive"]]["quantity"].sum()) / gross, 1),
    }


def _pump(trader, start, end, trades, px, flags) -> dict:
    span = trades[trades["tick"].between(start, end)]
    legs = _account_legs(span, trader)
    if legs.empty:
        return {"trades": 0}
    signed = np.where(legs["side"] == "buy", legs["quantity"], -legs["quantity"])
    path = pd.Series(np.cumsum(signed), index=legs["tick"].to_numpy())
    peak_tick = int(path.idxmax())
    before, after = legs[legs["tick"] <= peak_tick], legs[legs["tick"] > peak_tick]
    buys_before = before[before["side"] == "buy"]
    sells_after = after[after["side"] == "sell"]

    burst = {}
    agg_buys = legs[(legs["side"] == "buy") & legs["aggressive"]]
    if not agg_buys.empty:
        ticks = range(start, end + 1)
        push = agg_buys.groupby("tick")["quantity"].sum().reindex(ticks, fill_value=0).rolling(BURST_WINDOW).sum()
        mkt = span.groupby("tick")["quantity"].sum().reindex(ticks, fill_value=0).rolling(BURST_WINDOW).sum()
        if push.notna().any() and push.max() > 0:
            t_end = int(push.idxmax())
            t_start = t_end - BURST_WINDOW + 1
            burst = {
                "aggressive_buy_burst_ticks": [t_start, t_end],
                "burst_aggressive_buy_volume": int(push.max()),
                "burst_share_of_market_volume_pct": round(100 * float(push.max()) / float(mkt.loc[t_end]), 1) if mkt.loc[t_end] else 0.0,
                "price_change_over_burst_bp": _bp(px.get(t_start - 1, np.nan), px.get(t_end, np.nan)),
            }
    peers = []
    if flags is not None and not flags.empty:
        f = flags[flags["final_flag"] & (flags["predicted_persona"] == "pump_and_dump")
                  & (flags["window_end"] >= start) & (flags["window_start"] <= end) & (flags["trader_id"] != trader)]
        peers = sorted(f["trader_id"].unique().tolist())
    return {
        "trades": int(len(legs)),
        "bought_until_peak_position": int(buys_before["quantity"].sum()),
        "of_which_aggressive_buys": int(buys_before[buys_before["aggressive"]]["quantity"].sum()),
        "peak_position": int(path.max()),
        "peak_position_tick": peak_tick,
        "sold_after_peak": int(sells_after["quantity"].sum()),
        "of_which_aggressive_sells": int(sells_after[sells_after["aggressive"]]["quantity"].sum()),
        "end_position": int(path.iloc[-1]),
        "price_at_start": round(float(px.get(start, np.nan)), 4),
        "price_at_peak_position": round(float(px.get(peak_tick, np.nan)), 4),
        "price_at_end": round(float(px.get(end, np.nan)), 4),
        "price_change_start_to_peak_bp": _bp(px.get(start, np.nan), px.get(peak_tick, np.nan)),
        "price_change_peak_to_end_bp": _bp(px.get(peak_tick, np.nan), px.get(end, np.nan)),
        **burst,
        "other_accounts_flagged_for_pump_and_dump_in_same_span": peers[:8],
    }


def _generic(trader, start, end, trades, orders) -> dict:
    acct = orders[(orders["trader_id"] == trader) & orders["tick"].between(start, end)]
    placed = acct[acct["event_type"] == "placed"]
    legs = _account_legs(trades[trades["tick"].between(start, end)], trader)
    pv = float(placed["quantity"].sum())
    cv = float(acct[acct["event_type"] == "cancelled"]["quantity"].sum())
    return {
        "orders_placed": int(len(placed)),
        "median_order_size": int(placed["quantity"].median()) if len(placed) else 0,
        "largest_order": int(placed["quantity"].max()) if len(placed) else 0,
        "cancelled_volume_pct": round(100 * cv / pv, 1) if pv else 0.0,
        "trades": int(len(legs)),
        "bought": int(legs[legs["side"] == "buy"]["quantity"].sum()) if len(legs) else 0,
        "sold": int(legs[legs["side"] == "sell"]["quantity"].sum()) if len(legs) else 0,
    }


def build_case_evidence(
    persona: str,
    trader_id: str,
    commodity: str,
    start_tick: int,
    end_tick: int,
    trades: pd.DataFrame,
    orders: pd.DataFrame,
    flags: pd.DataFrame | None = None,
) -> dict:
    """Behavioural evidence for one case (selected market's logs). Never raises on sparse data."""
    t = trades[trades["commodity"] == commodity] if "commodity" in trades else trades
    o = orders[orders["commodity"] == commodity] if "commodity" in orders else orders
    if "quantity" in o:
        o = o.assign(quantity=pd.to_numeric(o["quantity"], errors="coerce"))
    px = _price_path(t)
    start, end = int(start_tick), int(end_tick)
    try:
        if persona == "spoofing":
            facts = _spoofing(trader_id, start, end, t, o, px)
        elif persona == "wash_trading":
            facts = _wash(trader_id, start, end, t)
        elif persona == "pump_and_dump":
            fl = flags[flags["commodity"] == commodity] if flags is not None and "commodity" in flags else flags
            facts = _pump(trader_id, start, end, t, px, fl)
        else:
            facts = _generic(trader_id, start, end, t, o)
    except Exception as exc:  # evidence is an aid; never break the case view
        facts = {"note": f"evidence unavailable: {type(exc).__name__}"}
    facts = {k: v for k, v in facts.items() if v is not None and not (isinstance(v, float) and v != v)}
    return {"span_ticks": [start, end], "span_length_ticks": end - start, **facts}

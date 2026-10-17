"""
api/market.py
=============
Market Overview aggregations: everything the browser draws is reduced here
(hundreds of points per series, never the raw ~0.5M-row trade log).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from api import data
from api.data import clean

INDEX_BUCKET = 50      # ticks per point on the cross-asset indexed chart
BAR_TICKS = 25         # candlestick bar size (ticks = simulated seconds)
FLOW_BUCKET = 60       # 1-minute buckets for volume / microstructure series
HEAT_BUCKET = 300      # 5-minute buckets for the trader heatmap
RETURN_TICKS = 30      # return horizon for the distribution chart


def _canon(p: str) -> str:
    p = str(p)
    for key, name in (("spoof", "spoofing"), ("wash", "wash_trading"), ("pump", "pump_and_dump"), ("layer", "layering")):
        if key in p:
            return name
    return p


def _events_list(ev: pd.DataFrame) -> list[dict]:
    if ev.empty:
        return []
    out = []
    for _, e in ev.iterrows():
        out.append({"event_id": e["event_id"], "persona": _canon(e["persona"]), "commodity": e["commodity"],
                    "start_tick": int(e["start_tick"]), "end_tick": int(e["end_tick"]),
                    "trader_ids": list(e["trader_ids"]) if e["trader_ids"] is not None else []})
    return out


def _kpi(t: pd.DataFrame, ev: pd.DataFrame) -> dict:
    traders = set(t["buyer_trader_id"].dropna().unique()) | set(t["seller_trader_id"].dropna().unique()) if not t.empty else set()
    return {"trades": int(len(t)), "volume": int(t["quantity"].sum()) if not t.empty else 0,
            "traders": len(traders), "events": int(len(ev))}


def _persona_f1(metrics: dict) -> list[dict]:
    return [{"persona": p, "f1": m.get("f1"), "precision": m.get("precision"), "recall": m.get("recall"), "fpr": m.get("fpr")}
            for p, m in (metrics.get("personas") or {}).items()]


def overview(run: str, rnd: int, market: str | None) -> dict:
    t = data.trades(run, rnd, market)
    ev = data.events(run, rnd, market)
    if t.empty:
        return {"commodities": [], "kpi": _kpi(t, ev), "snapshot": []}
    commodities = sorted(t["commodity"].unique().tolist())

    # last traded price per bucket, per commodity -> indexed to 100 at the first bucket
    b = t.assign(bucket=(t["tick"] // INDEX_BUCKET) * INDEX_BUCKET)
    px = b.groupby(["bucket", "commodity"])["price"].last().unstack("commodity").sort_index().ffill().bfill()
    indexed = px / px.iloc[0] * 100.0
    snapshot = []
    for c in commodities:
        ct = t[t["commodity"] == c]
        first, last = float(px[c].iloc[0]), float(px[c].iloc[-1])
        snapshot.append({"commodity": c, "first": first, "last": last, "pct": (last / first - 1) * 100 if first else 0.0,
                         "volume": int(ct["quantity"].sum()), "trades": int(len(ct)),
                         "events": int((ev["commodity"] == c).sum()) if not ev.empty else 0,
                         "spark": [round(float(v), 4) for v in px[c].to_numpy()]})

    rets = px.pct_change().dropna()
    corr = rets.corr().fillna(0.0)
    by_persona = (ev.assign(persona=ev["persona"].map(_canon)).groupby(["commodity", "persona"]).size().unstack(fill_value=0)
                  .reindex(commodities).fillna(0) if not ev.empty else pd.DataFrame(index=commodities))
    return clean({
        "commodities": commodities,
        "kpi": _kpi(t, ev),
        "persona_f1": _persona_f1(data.metrics(run, rnd)),
        "snapshot": snapshot,
        "indexed": {"ticks": [int(x) for x in indexed.index], "series": {c: [round(float(v), 3) for v in indexed[c]] for c in commodities}},
        "prices": {"ticks": [int(x) for x in px.index], "series": {c: [round(float(v), 4) for v in px[c]] for c in commodities}},
        "correlation": {"labels": commodities, "matrix": [[round(float(v), 3) for v in row] for row in corr.reindex(index=commodities, columns=commodities).to_numpy()]},
        "volume_by_commodity": [{"commodity": s["commodity"], "volume": s["volume"], "trades": s["trades"]} for s in snapshot],
        "events_by_commodity": [{"commodity": c, **{p: int(by_persona.loc[c, p]) if p in by_persona.columns else 0 for p in data.PERSONAS}}
                                for c in commodities],
        "events": _events_list(ev),
    })


def commodity(run: str, rnd: int, market: str | None, comm: str) -> dict:
    t_all = data.trades(run, rnd, market)
    if t_all.empty or comm not in set(t_all["commodity"]):
        return {"commodity": comm, "empty": True}
    t = t_all[t_all["commodity"] == comm].sort_values("tick")
    ev_all = data.events(run, rnd, market)
    ev = ev_all[ev_all["commodity"] == comm] if not ev_all.empty else ev_all

    # candles from trades
    g = t.assign(bar=(t["tick"] // BAR_TICKS) * BAR_TICKS).groupby("bar")
    pv = (t["price"] * t["quantity"]).groupby((t["tick"] // BAR_TICKS) * BAR_TICKS).sum()
    candles = pd.DataFrame({"open": g["price"].first(), "high": g["price"].max(), "low": g["price"].min(),
                            "close": g["price"].last(), "volume": g["quantity"].sum()})
    candles["vwap"] = pv / candles["volume"]
    candles = candles.reset_index().rename(columns={"bar": "tick"})

    # aggressive buy / sell volume and imbalance per minute
    f = t.assign(bucket=(t["tick"] // FLOW_BUCKET) * FLOW_BUCKET)
    buy = f[f["aggressor_side"] == "buy"].groupby("bucket")["quantity"].sum()
    sell = f[f["aggressor_side"] == "sell"].groupby("bucket")["quantity"].sum()
    n_tr = f.groupby("bucket").size()
    idx = n_tr.index
    buy, sell = buy.reindex(idx, fill_value=0), sell.reindex(idx, fill_value=0)

    # order-log microstructure per minute
    o = data.order_events(run, rnd, market)
    o = o[o["commodity"] == comm] if not o.empty else o
    if not o.empty:
        ob = (o["tick"] // FLOW_BUCKET) * FLOW_BUCKET
        placed = o[o["event_type"] == "placed"].groupby(ob[o["event_type"] == "placed"]).size().reindex(idx, fill_value=0)
        cancelled = o[o["event_type"] == "cancelled"].groupby(ob[o["event_type"] == "cancelled"]).size().reindex(idx, fill_value=0)
    else:
        placed = cancelled = pd.Series(0, index=idx)
    micro = pd.DataFrame({
        "tick": idx, "buy": buy.to_numpy(), "sell": sell.to_numpy(), "trades": n_tr.to_numpy(),
        "imbalance": ((buy - sell) / (buy + sell).replace(0, np.nan)).fillna(0).to_numpy(),
        "cancel_rate": (cancelled / (placed + cancelled).replace(0, np.nan)).fillna(0).to_numpy(),
        "otr": (placed / n_tr.replace(0, np.nan)).fillna(0).to_numpy(),
        "placed": placed.to_numpy(), "cancelled": cancelled.to_numpy(),
    })

    # volume profile by price level
    bins = pd.cut(t["price"], bins=40)
    prof = t.groupby(bins, observed=False)["quantity"].sum()
    profile = [{"price": round(float(iv.mid), 4), "volume": int(v)} for iv, v in prof.items()]

    # 30-second returns (bp) of the last traded price, with a normal fit. Tick-to-tick
    # returns sit on the contract's price grid and give a comb-like histogram.
    last = t.groupby("tick")["price"].last()
    last = last.reindex(range(int(last.index.min()), int(last.index.max()) + 1)).ffill()
    r = (np.log(last.iloc[::RETURN_TICKS]).diff().dropna() * 1e4)
    hist = None
    if len(r) > 20:
        lo, hi = np.percentile(r, [0.5, 99.5])
        counts, edges = np.histogram(r.clip(lo, hi), bins=40, density=True)
        mu, sd = float(r.mean()), float(r.std())
        centers = (edges[:-1] + edges[1:]) / 2
        normal = np.exp(-0.5 * ((centers - mu) / sd) ** 2) / (sd * np.sqrt(2 * np.pi)) if sd > 0 else np.zeros_like(centers)
        kurt = float(((r - mu) ** 4).mean() / sd ** 4 - 3) if sd > 0 else 0.0
        hist = {"centers": centers.round(3).tolist(), "density": counts.round(5).tolist(), "normal": normal.round(5).tolist(),
                "mean": mu, "std": sd, "excess_kurtosis": kurt, "n": int(len(r))}

    # top-20 trader activity heatmap
    h = t.assign(bucket=(t["tick"] // HEAT_BUCKET) * HEAT_BUCKET)
    side = pd.concat([h[["buyer_trader_id", "bucket", "quantity"]].rename(columns={"buyer_trader_id": "trader"}),
                      h[["seller_trader_id", "bucket", "quantity"]].rename(columns={"seller_trader_id": "trader"})])
    vol = side.groupby(["trader", "bucket"])["quantity"].sum()
    top = side.groupby("trader")["quantity"].sum().nlargest(20).index.tolist()
    piv = vol.unstack("bucket").reindex(top).fillna(0)
    heat = {"traders": top, "buckets": [int(x) for x in piv.columns], "values": piv.to_numpy().astype(int).tolist()}

    first, last_p = float(t["price"].iloc[0]), float(t["price"].iloc[-1])
    return clean({
        "commodity": comm,
        "kpi": {**_kpi(t, ev), "last": last_p, "first": first, "pct": (last_p / first - 1) * 100 if first else 0.0,
                "high": float(t["price"].max()), "low": float(t["price"].min())},
        "persona_f1": _persona_f1(data.metrics(run, rnd)),
        "events": _events_list(ev),
        "candles": data.records(candles),
        "flow": data.records(micro),
        "profile": profile,
        "returns": hist,
        "heatmap": heat,
        "bucket_ticks": {"bar": BAR_TICKS, "flow": FLOW_BUCKET, "heat": HEAT_BUCKET},
    })

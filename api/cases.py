"""
api/cases.py
============
Surveillance (Flagged Trades) views: detector flags merged into cases, matched
against the simulator's ground truth, plus per-case evidence, SHAP, price
context, raw orders/trades and the fact-checked AI case note.
Logic ported from dashboard/components/tab_flags.py.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from api import data
from api.data import clean

logger = logging.getLogger(__name__)

FAKE_DEPTH = {"spoofing", "layering"}
CASE_PAD = 120


def case_id(c: dict) -> str:
    return f"{c['commodity']}~{c['trader_id']}~{c['persona']}~{int(c['start_tick'])}~{int(c['end_tick'])}"


def parse_case_id(cid: str) -> dict:
    try:
        comm, trader, persona, start, end = cid.split("~")
        return {"commodity": comm, "trader_id": trader, "persona": persona, "start_tick": int(start), "end_tick": int(end)}
    except ValueError as exc:
        from fastapi import HTTPException
        raise HTTPException(400, "invalid case id") from exc


def _build_cases(flags: pd.DataFrame) -> list[dict]:
    """Merge consecutive flagged windows of the same account + persona into cases."""
    f = flags[flags["final_flag"]]
    if f.empty:
        return []
    f = f.sort_values(["commodity", "trader_id", "predicted_persona", "window_start"])
    cases: list[dict] = []
    for (comm, trader, persona), g in f.groupby(["commodity", "trader_id", "predicted_persona"], sort=False):
        cur = None
        for row in g.itertuples(index=False):
            if cur is not None and row.window_start <= cur["end_tick"] + 1:
                cur["end_tick"] = max(cur["end_tick"], int(row.window_end))
                cur["n_windows"] += 1
                cur["triggered_by"] |= set(row.triggered_by)
                if row.final_score > cur["score"]:
                    cur.update(score=float(row.final_score), top_shap_features=row.top_shap_features,
                               persona_scores=getattr(row, "persona_scores", None) or {},
                               iso_score=float(getattr(row, "iso_score", 0.0)))
                continue
            if cur is not None:
                cases.append(cur)
            cur = {"commodity": comm, "trader_id": trader, "persona": persona,
                   "start_tick": int(row.window_start), "end_tick": int(row.window_end),
                   "score": float(row.final_score), "n_windows": 1, "triggered_by": set(row.triggered_by),
                   "top_shap_features": row.top_shap_features,
                   "persona_scores": getattr(row, "persona_scores", None) or {},
                   "iso_score": float(getattr(row, "iso_score", 0.0))}
        if cur is not None:
            cases.append(cur)
    for c in cases:
        c["triggered_by"] = sorted(c["triggered_by"])
        c["id"] = case_id(c)
    return sorted(cases, key=lambda c: -c["score"])


def _match(cases: list[dict], ev: pd.DataFrame) -> list[dict]:
    """Attach the overlapping ground-truth episode to each case; return the missed episodes."""
    matched: set[str] = set()
    evs = [] if ev.empty else ev.to_dict("records")
    for c in cases:
        c["gt_event"] = c["gt_persona"] = None
        for e in evs:
            if e["commodity"] != c["commodity"] or c["trader_id"] not in set(e["trader_ids"]):
                continue
            if c["end_tick"] < e["start_tick"] or c["start_tick"] > e["end_tick"]:
                continue
            c["gt_event"], c["gt_persona"] = e["event_id"], e["persona"]
            if c["persona"] == e["persona"]:
                matched.add(e["event_id"])
            break
        c["status"] = ("no_episode" if c["gt_event"] is None
                       else "confirmed" if c["gt_persona"] == c["persona"] else "other_persona")
    return [{"event_id": e["event_id"], "persona": e["persona"], "commodity": e["commodity"],
             "start_tick": int(e["start_tick"]), "end_tick": int(e["end_tick"]), "trader_ids": list(e["trader_ids"])}
            for e in evs if e["event_id"] not in matched]


@data.ttl_cache(300, maxsize=16)
def _cases_cached(run: str, rnd: int, market: str | None) -> tuple[list[dict], list[dict], dict]:
    lean = data.flags_lean_for_market(run, rnd, market)
    ev = data.events(run, rnd, market)
    if lean.empty:
        return [], [], {"has_flags": False}
    flags = data.flags_for_market(run, rnd, market)
    cases = _build_cases(flags) if not flags.empty else []
    missed = _match(cases, ev)
    meta = {"has_flags": True,
            "review_windows": int((lean["alert_tier"] == "review").sum()) if "alert_tier" in lean.columns else 0,
            "windows": int(len(lean)), "alert_windows": int(lean["final_flag"].sum())}
    return cases, missed, meta


def summary(run: str, rnd: int, market: str | None) -> dict:
    cases, missed, meta = _cases_cached(run, rnd, market)
    cls = [c for c in cases if c["persona"] != "anomaly"]
    anom = [c for c in cases if c["persona"] == "anomaly"]
    kpi = {
        "persona_cases": len(cls),
        "confirmed": sum(c["status"] == "confirmed" for c in cls),
        "no_episode": sum(c["status"] == "no_episode" for c in cls),
        "other_persona": sum(c["status"] == "other_persona" for c in cls),
        "missed": len(missed),
        "anomalies": len(anom),
        "anomalies_on_episode": sum(c["gt_event"] is not None for c in anom),
        "sibling_label": sum(c["gt_persona"] in FAKE_DEPTH and c["persona"] in FAKE_DEPTH and c["gt_persona"] != c["persona"]
                             for c in cls),
        **{k: meta.get(k, 0) for k in ("review_windows", "windows", "alert_windows")},
    }
    slim = [{k: c[k] for k in ("id", "commodity", "trader_id", "persona", "start_tick", "end_tick", "score", "n_windows",
                               "triggered_by", "status", "gt_event", "gt_persona")} for c in cases]
    return clean({"has_flags": meta.get("has_flags", False), "kpi": kpi, "cases": slim, "missed": missed,
                  "thresholds": data.thresholds(run, rnd)})


def _find(run: str, rnd: int, market: str | None, cid: str) -> dict:
    cases, _, _ = _cases_cached(run, rnd, market)
    for c in cases:
        if c["id"] == cid:
            return c
    from fastapi import HTTPException
    raise HTTPException(404, "case not found")


@data.ttl_cache(600, maxsize=64)
def _evidence(run: str, rnd: int, market: str | None, cid: str) -> dict:
    from src.agentic.case_evidence import build_case_evidence
    c = parse_case_id(cid)
    persona = c["persona"]
    trades = data.trades(run, rnd, market)
    if persona in ("spoofing", "layering", "anomaly"):
        orders = data.orders(run, rnd, market)
    else:
        orders = pd.DataFrame(columns=["order_id", "trader_id", "side", "price", "quantity", "tick", "event_type", "commodity"])
    flags = data.flags(run, rnd) if persona == "pump_and_dump" else None
    try:
        return build_case_evidence(persona, c["trader_id"], c["commodity"], c["start_tick"], c["end_tick"], trades, orders, flags)
    except Exception as exc:   # evidence is supporting detail; never fail the case view on it
        logger.warning("case evidence failed for %s: %s", cid, exc)
        return {}


def detail(run: str, rnd: int, market: str | None, cid: str) -> dict:
    c = _find(run, rnd, market, cid)
    t = data.trades(run, rnd, market)
    w = t[(t["commodity"] == c["commodity"]) & t["tick"].between(c["start_tick"] - CASE_PAD, c["end_tick"] + CASE_PAD)]
    px = w.groupby("tick")["price"].last()
    mine = w[(w["buyer_trader_id"] == c["trader_id"]) | (w["seller_trader_id"] == c["trader_id"])]
    buys = mine[mine["buyer_trader_id"] == c["trader_id"]]
    sells = mine[mine["seller_trader_id"] == c["trader_id"]]
    shap = {k: float(v) for k, v in (c.get("top_shap_features") or {}).items() if v is not None}
    return clean({
        **{k: c[k] for k in ("id", "commodity", "trader_id", "persona", "start_tick", "end_tick", "score", "n_windows",
                             "triggered_by", "status", "gt_event", "gt_persona", "iso_score")},
        "persona_scores": c.get("persona_scores") or {},
        "thresholds": data.thresholds(run, rnd),
        "shap": sorted(({"feature": k, "value": v} for k, v in shap.items()), key=lambda d: -abs(d["value"])),
        "evidence": {k: v for k, v in _evidence(run, rnd, market, cid).items() if k != "span_ticks"},
        "price": {"ticks": [int(x) for x in px.index], "prices": [float(v) for v in px.to_numpy()]},
        "buys": [{"tick": int(r.tick), "price": float(r.price), "qty": int(r.quantity)} for r in buys.itertuples()],
        "sells": [{"tick": int(r.tick), "price": float(r.price), "qty": int(r.quantity)} for r in sells.itertuples()],
    })


def raw(run: str, rnd: int, market: str | None, cid: str, limit: int = 500) -> dict:
    c = _find(run, rnd, market, cid)
    o = data.orders(run, rnd, market)
    t = data.trades(run, rnd, market)
    o = o[(o["commodity"] == c["commodity"]) & (o["trader_id"] == c["trader_id"]) & o["tick"].between(c["start_tick"], c["end_tick"])]
    t = t[(t["commodity"] == c["commodity"])
          & ((t["buyer_trader_id"] == c["trader_id"]) | (t["seller_trader_id"] == c["trader_id"]))
          & t["tick"].between(c["start_tick"], c["end_tick"])]
    return {"orders_total": int(len(o)), "trades_total": int(len(t)),
            "orders": data.records(o.head(limit)), "trades": data.records(t.head(limit))}


def narrative(run: str, rnd: int, market: str | None, cid: str) -> dict:
    from src.agentic.llm_client import llm_unavailable_reason
    cfg = data.llm_config("narrative_generator")
    reason = llm_unavailable_reason(cfg)
    if reason:
        return {"error": reason}
    c = _find(run, rnd, market, cid)
    try:
        from src.agentic.case_evidence import MECHANISMS
        from src.agentic.narrative_generator import NarrativeGenerator
        gen = NarrativeGenerator(config=cfg, run_dir=data.run_dir(run))
        flag_row = {"trader_id": c["trader_id"], "commodity": c["commodity"], "flagged_persona": c["persona"],
                    "window_start_tick": int(c["start_tick"]), "window_end_tick": int(c["end_tick"]),
                    "detector_score": round(float(c["score"]), 3), "triggered_by": list(c["triggered_by"])}
        shap = {k: round(float(v), 3) for k, v in (c.get("top_shap_features") or {}).items() if v is not None}
        res = gen.generate(flag_row=flag_row, shap_summary=shap, evidence=_evidence(run, rnd, market, cid),
                           mechanism=MECHANISMS.get(c["persona"]))
        return clean({"narrative_text": res.get("narrative_text"), "grounding_check_passed": bool(res.get("grounding_check_passed"))})
    except Exception as exc:
        logger.warning("narrative generation failed: %s", exc)
        return {"error": str(exc)}

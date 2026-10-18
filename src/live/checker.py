"""
src/live/checker.py
===================
Checks a flagged window against how each manipulation persona actually moves a
market, using only what anonymous 1-minute bars can show. Real data has no
ground truth, so this is the second opinion on every model flag.

For each window it measures the evidence once (move, volume, range, wicks, how
much of the move was given back, whether other commodities moved with it, and
which scheduled events it overlaps), then tests each persona's mechanism:

  pump_and_dump  sharp run-up on rising volume, then most of it given back, in one
                 book only (a sector-wide move is news, not a pump)
  wash_trading   heavy volume that does not move the price (no price discovery),
                 away from settlements, opens and data releases
  spoofing       a price push on light volume that snaps back within minutes, with
                 long wicks (the order book moved the price, trades did not)
  layering       like spoofing but a slower, stepped push; also needs the order book

Verdicts:
  consistent        every mechanism test passes and nothing ordinary explains it
  partial           some tests pass
  likely benign     the pattern is there but a scheduled event or a market-wide move explains it
  not consistent    the window does not behave like this persona
Spoofing, layering and wash trading can at best be "consistent (unverifiable)":
proving them needs order-book or account data that public bars do not have.

An optional LLM reviewer reads the same evidence and gives a written opinion;
it never sees anything the rules did not measure.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

import numpy as np
import pandas as pd

from src.live.bars import minute_to_utc
from src.live.calendar import events_overlapping
from src.live.datasets import SECTOR

logger = logging.getLogger(__name__)

PRE_MIN, POST_MIN = 60, 30          # baseline before the window, look for a reversal after it
SIGMA_FLOOR_BP = 0.5
MIN_JUDGED_Z = 1.0                  # give-back / retrace shares are only meaningful after a real move

RULES = {
    "run_up_z": 2.5, "pd_volume_ratio": 1.5, "giveback": 0.5,
    "wash_volume_ratio": 2.5, "wash_max_move_z": 1.0, "wash_max_range_ratio": 1.3,
    "spoof_push_z": 2.0, "spoof_max_volume_ratio": 1.3, "spoof_retrace": 0.6, "spoof_wick": 0.6,
    "layer_push_z": 1.5, "layer_min_minutes": 3,
    "comove_z": 2.0, "thin_zero_volume": 0.3, "min_post_bars": 10,
}

DATA_NEEDED = {
    "spoofing": "Full order book with order ids (placements and cancels by price level) to show large orders "
                "placed and pulled without intent to trade; account ids to tie them to the opposite-side fills.",
    "layering": "Order book depth by price level over time, showing orders stacked on several levels and cancelled "
                "together after a fill; account ids.",
    "wash_trading": "Trade records with buyer and seller account ids (or beneficial owners) to show the same party "
                    "or colluding accounts on both sides.",
    "pump_and_dump": "Account-level positions over time (accumulation, then selling into the rally) and any "
                     "promotion or coordination evidence.",
}
VERDICT_ORDER = ["consistent", "consistent (unverifiable)", "partial", "likely benign", "not consistent", "insufficient data"]


def _pct(x) -> str:
    return "n/a, move too small to judge" if x is None else f"{int(round(x))}%"


def _r(x, n=2):
    return None if x is None or (isinstance(x, float) and not np.isfinite(x)) else round(float(x), n)


class MechanismChecker:
    """Rule-based persona mechanism tests on a flagged window of 1-minute bars."""

    def __init__(self, bars: pd.DataFrame, *, real_time: bool = True, rules: dict | None = None) -> None:
        self.rules = {**RULES, **(rules or {})}
        self.real_time = real_time
        self.series = {c: b.set_index("minute").sort_index() for c, b in bars.groupby("commodity")}

    # ------------------------------------------------------------------ evidence
    def _move(self, c: str, s: int, e: int) -> dict | None:
        b = self.series.get(c)
        if b is None:
            return None
        win = b.loc[s:e]
        if win.empty:
            return None
        pre = b.loc[s - PRE_MIN:s - 1]
        ref = float(pre["close"].iloc[-1]) if len(pre) else float(win["open"].iloc[0])
        ret_pre = np.log(pre["close"]).diff().dropna() * 1e4
        sigma = max(float(ret_pre.std()) if len(ret_pre) > 5 else float((np.log(win["close"]).diff().dropna() * 1e4).std() or 0),
                    SIGMA_FLOOR_BP)
        n = len(win)
        path = np.log(win["close"].to_numpy() / ref) * 1e4
        return {"b": b, "win": win, "pre": pre, "ref": ref, "sigma": sigma, "n": n, "path": path,
                "move_bp": float(path[-1]), "move_z": float(path[-1] / (sigma * np.sqrt(n)))}

    def evidence(self, commodity: str, start: int, end: int) -> dict[str, Any]:
        m = self._move(commodity, start, end)
        if m is None:
            return {"available": False}
        b, win, pre, sigma, n, path = m["b"], m["win"], m["pre"], m["sigma"], m["n"], m["path"]
        post = b.loc[end + 1:end + POST_MIN]
        k = int(np.argmax(np.abs(path)))
        peak = float(path[k])
        scale = sigma * np.sqrt(max(k + 1, 1))
        end_close = float(win["close"].iloc[-1])
        post_move = float(np.log(post["close"].iloc[-1] / end_close) * 1e4) if len(post) else None
        post5 = b.loc[end + 1:end + 5]
        snap = float(np.log(post5["close"].iloc[-1] / end_close) * 1e4) if len(post5) else 0.0
        vol_base = max(float(pre["volume"].median()) if len(pre) else float(win["volume"].median()), 1.0)
        rng = (win["high"] - win["low"]) / win["close"] * 1e4
        rng_base = max(float(((pre["high"] - pre["low"]) / pre["close"] * 1e4).median()) if len(pre) else float(rng.median()), 0.1)
        span = (win["high"] - win["low"]).to_numpy(float)
        body = (win["close"] - win["open"]).abs().to_numpy(float)
        wick = float(np.mean(np.where(span > 0, 1 - body / np.where(span > 0, span, 1), 0.0)))
        # Share of the peak excursion undone by 5 minutes after the window (1 = fully back to the start)
        final = m["move_bp"] + snap
        retrace = 1.0 - (final * np.sign(peak)) / abs(peak) if abs(peak / scale) >= MIN_JUDGED_Z else None
        giveback = (-post_move / m["move_bp"]) if (post_move is not None and abs(m["move_z"]) >= MIN_JUDGED_Z) else None
        # Same-window moves in the other books
        peers = []
        for c in self.series:
            if c == commodity:
                continue
            pm = self._move(c, start, end)
            if pm is None:
                continue
            same_sector = SECTOR.get(c) == SECTOR.get(commodity)
            if abs(pm["move_z"]) >= self.rules["comove_z"] and np.sign(pm["move_z"]) == np.sign(m["move_z"]):
                peers.append({"commodity": c, "move_z": _r(pm["move_z"]), "same_sector": same_sector})
        ev = {
            "available": True,
            "minutes": n,
            "pre_bars": int(len(pre)),
            "post_bars": int(len(post)),
            "sigma_1min_bp": _r(sigma),
            "move_bp": _r(m["move_bp"]),
            "move_z": _r(m["move_z"]),
            "peak_excursion_bp": _r(peak),
            "peak_excursion_z": _r(peak / scale),
            "peak_minute_offset": k,
            "max_run_up_z": _r(max(path.max(), 0) / (sigma * np.sqrt(n))),
            "retrace_pct_by_5min_after": None if retrace is None else _r(float(np.clip(retrace, -1, 1)) * 100, 0),
            "post_30min_move_bp": _r(post_move),
            # % of the window's move reversed in the next 30 min (negative = kept going); capped at +-300
            "giveback_pct": None if giveback is None else _r(float(np.clip(giveback, -3, 3)) * 100, 0),
            "volume_ratio": _r(float(win["volume"].mean()) / vol_base),
            "volume_ratio_after": _r(float(post["volume"].mean()) / vol_base) if len(post) else None,
            "range_ratio": _r(float(rng.mean()) / rng_base),
            "wick_share": _r(wick),
            "zero_volume_share": _r(float((win["volume"] <= 0).mean())),
            "zero_volume_share_before": _r(float((pre["volume"] <= 0).mean())) if len(pre) else None,
            "comoving_commodities": peers,
            "n_comoving": len(peers),
            "n_comoving_same_sector": sum(p["same_sector"] for p in peers),
        }
        if self.real_time:
            s_utc, e_utc = minute_to_utc([start])[0], minute_to_utc([end])[0] + pd.Timedelta(minutes=1)
            ev["start_utc"], ev["end_utc"] = s_utc.strftime("%Y-%m-%d %H:%M"), e_utc.strftime("%Y-%m-%d %H:%M")
            ev["start_new_york"] = s_utc.tz_convert("America/New_York").strftime("%a %H:%M %Z")
            ev["scheduled_events"] = events_overlapping(commodity, s_utc, e_utc)
        else:
            ev["scheduled_events"] = []
        return ev

    # ------------------------------------------------------------------ tests
    def _benign(self, ev: dict) -> list[str]:
        out = []
        fixed = [e for e in ev["scheduled_events"] if e["certainty"] == "fixed"]
        possible = [e for e in ev["scheduled_events"] if e["certainty"] == "possible"]
        if fixed:
            out.append("overlaps " + "; ".join(f"{e['event']} ({e['at_local']})" for e in fixed))
        if possible:
            out.append("falls in a usual release window: " + "; ".join(e["event"] for e in possible))
        if ev["n_comoving_same_sector"] >= 1 or ev["n_comoving"] >= 2:
            names = ", ".join(p["commodity"] for p in ev["comoving_commodities"])
            out.append(f"moved together with {names} (a sector or market-wide move points to news, not one manipulator)")
        if (ev.get("zero_volume_share") or 0) >= self.rules["thin_zero_volume"]:
            out.append("thin market: many minutes with no trades, so small orders move the price")
        return out

    def _verdict(self, tests: list[tuple[str, bool]], benign: list[str], unverifiable: bool, hard_benign: bool) -> tuple[str, float]:
        passed = sum(ok for _, ok in tests)
        share = passed / len(tests)
        if share == 1.0 and hard_benign:
            return "likely benign", 0.2
        if share == 1.0:
            return ("consistent (unverifiable)" if unverifiable else "consistent"), (0.5 if unverifiable else 0.75) * (0.7 if benign else 1.0)
        if share >= 0.5:
            return ("likely benign" if hard_benign else "partial"), round(0.3 * share, 2)
        return "not consistent", round(0.1 * share, 2)

    def check(self, commodity: str, start: int, end: int) -> dict[str, Any]:
        ev = self.evidence(commodity, start, end)
        if not ev.get("available"):
            return {"evidence": ev, "personas": {}, "best_match": None}
        R = self.rules
        benign = self._benign(ev)
        hard_benign = any(e["certainty"] == "fixed" for e in ev["scheduled_events"]) or ev["n_comoving_same_sector"] >= 1 \
            or ev["n_comoving"] >= 2
        res = {}
        enough_post = ev["post_bars"] >= R["min_post_bars"]

        # Pump and dump
        gb = ev["giveback_pct"]
        tests = [
            (f"run-up of at least {R['run_up_z']} sigma (max run-up {ev['max_run_up_z']} sigma)", (ev["max_run_up_z"] or 0) >= R["run_up_z"]),
            (f"volume at least {R['pd_volume_ratio']}x normal during the run-up ({ev['volume_ratio']}x)", (ev["volume_ratio"] or 0) >= R["pd_volume_ratio"]),
            (f"at least {int(R['giveback'] * 100)}% of the move given back within {POST_MIN} min "
             f"({'n/a, too few bars after the window' if not enough_post else _pct(gb)})",
             enough_post and gb is not None and gb >= R["giveback"] * 100),
            ("price went up (a pump)", (ev["move_bp"] or 0) > 0 or (ev["peak_excursion_bp"] or 0) > 0),
        ]
        res["pump_and_dump"] = self._persona_result("pump_and_dump", tests, benign, False, hard_benign)

        # Wash trading
        tests = [
            (f"volume at least {R['wash_volume_ratio']}x normal ({ev['volume_ratio']}x)", (ev["volume_ratio"] or 0) >= R["wash_volume_ratio"]),
            (f"price barely moved (|move| {abs(ev['move_z'] or 0)} sigma, at most {R['wash_max_move_z']})", abs(ev["move_z"] or 0) <= R["wash_max_move_z"]),
            (f"bars stayed narrow (range {ev['range_ratio']}x normal, at most {R['wash_max_range_ratio']}x)", (ev["range_ratio"] or 9) <= R["wash_max_range_ratio"]),
        ]
        res["wash_trading"] = self._persona_result("wash_trading", tests, benign, True, hard_benign)

        # Spoofing
        tests = [
            (f"price pushed at least {R['spoof_push_z']} sigma (peak {abs(ev['peak_excursion_z'] or 0)} sigma)", abs(ev["peak_excursion_z"] or 0) >= R["spoof_push_z"]),
            (f"on light volume (at most {R['spoof_max_volume_ratio']}x normal; was {ev['volume_ratio']}x)", (ev["volume_ratio"] or 9) <= R["spoof_max_volume_ratio"]),
            (f"at least {int(R['spoof_retrace'] * 100)}% of the push undone within 5 min after "
             f"({_pct(ev['retrace_pct_by_5min_after'])})", (ev["retrace_pct_by_5min_after"] or 0) >= R["spoof_retrace"] * 100),
            (f"long wicks (wick share {ev['wick_share']}, at least {R['spoof_wick']})", (ev["wick_share"] or 0) >= R["spoof_wick"]),
        ]
        res["spoofing"] = self._persona_result("spoofing", tests, benign, True, hard_benign)

        # Layering
        tests = [
            (f"a stepped push of at least {R['layer_push_z']} sigma (peak {abs(ev['peak_excursion_z'] or 0)} sigma)", abs(ev["peak_excursion_z"] or 0) >= R["layer_push_z"]),
            (f"built over at least {R['layer_min_minutes']} minutes (peak at minute {ev['peak_minute_offset'] + 1})", ev["peak_minute_offset"] + 1 >= R["layer_min_minutes"]),
            (f"on light volume (at most {R['spoof_max_volume_ratio']}x normal; was {ev['volume_ratio']}x)", (ev["volume_ratio"] or 9) <= R["spoof_max_volume_ratio"]),
            (f"at least half of the push undone within 5 min after ({_pct(ev['retrace_pct_by_5min_after'])})", (ev["retrace_pct_by_5min_after"] or 0) >= 50),
        ]
        res["layering"] = self._persona_result("layering", tests, benign, True, hard_benign)

        best = min(res, key=lambda p: (VERDICT_ORDER.index(res[p]["verdict"]), -res[p]["confidence"]))
        return {"evidence": ev, "benign_explanations": benign, "personas": res, "best_match": best}

    def _persona_result(self, persona, tests, benign, unverifiable, hard_benign) -> dict:
        verdict, conf = self._verdict(tests, benign, unverifiable, hard_benign)
        return {
            "verdict": verdict,
            "confidence": round(float(conf), 2),
            "tests": [{"test": t, "passed": bool(ok)} for t, ok in tests],
            "passed": f"{sum(ok for _, ok in tests)}/{len(tests)}",
            "data_needed_to_confirm": DATA_NEEDED[persona],
        }


# ---------------------------------------------------------------------------
# LLM reviewer
# ---------------------------------------------------------------------------

LLM_SYSTEM = """You review windows of REAL commodity futures 1-minute bars that a machine-learning model flagged as possible market manipulation. The model was trained only on a market SIMULATOR and is weak on anonymous bars, so most flags are expected to be ordinary market activity.

You get measured evidence for one window (all numbers already computed), the rule-based test results for four manipulation types, and the scheduled market events it overlaps. The data has NO order book, NO cancels and NO account ids.

Mechanisms:
- pump_and_dump: one party accumulates, drives a sharp run-up (often with rising volume) to attract momentum traders, then sells into it; the price gives most of it back. One book only.
- wash_trading: the same or colluding accounts trade with each other to print volume without changing ownership; heavy volume with little price movement. Proof needs both sides' account ids.
- spoofing: large orders placed (and cancelled before they trade) on one side to push the price toward a small genuine order on the other side; price moves on light volume and snaps back. Proof needs the order book.
- layering: several orders stacked on consecutive price levels to fake depth, cancelled together after a fill; a stepped push that reverses. Proof needs the order book.

Ordinary explanations to weigh first: scheduled data releases, settlements, benchmark fixes, session opens/halts, sector-wide or market-wide moves (news, macro, the dollar), thin overnight liquidity, contract rolls, large legitimate hedging or execution orders.

Reply with ONLY a JSON object:
{"verdict": "consistent" | "partially consistent" | "likely benign" | "not consistent" | "cannot tell",
 "persona_assessed": "<the persona you think fits best, or none>",
 "confidence": "low" | "medium" | "high",
 "reasoning": "<3-5 sentences in plain language, citing the evidence numbers exactly as given>",
 "benign_explanations": ["..."],
 "data_needed": "<what data would settle it>"}
Never claim manipulation is proven. Use only numbers present in the input."""


_ASCII = str.maketrans({"‐": "-", "‑": "-", "‒": "-", "–": "-", "—": "-", "−": "-",
                        "‘": "'", "’": "'", "“": '"', "”": '"', " ": " ", " ": " ",
                        " ": " ", "≈": "~", "×": "x", "σ": "sigma"})


def _parse_json(text: str) -> dict | None:
    m = re.search(r"\{.*\}", text or "", re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


class LLMReviewer:
    """Optional written second opinion on a checked window (uses the project's budget-capped LLM client)."""

    def __init__(self, run_dir=None) -> None:
        from src.agentic.llm_client import LLMClient, LLMUnavailableError, llm_unavailable_reason, load_llm_settings

        cfg = load_llm_settings().get("live_checker", {})
        reason = llm_unavailable_reason(cfg)
        if reason:
            raise LLMUnavailableError(reason)
        self.client = LLMClient(cfg, run_dir=run_dir)

    def review(self, case: dict, check: dict) -> dict:
        ev = {k: v for k, v in check["evidence"].items() if k != "available"}
        payload = {
            "window": {k: case[k] for k in ("commodity", "start_utc", "end_utc", "minutes", "model_persona", "model_probability") if k in case},
            "evidence": ev,
            "rule_tests": {p: {"verdict": r["verdict"], "tests": r["tests"]} for p, r in check["personas"].items()},
            "benign_explanations_found": check.get("benign_explanations", []),
        }
        try:
            resp = self.client.complete(system=LLM_SYSTEM, messages=[{"role": "user", "content": json.dumps(payload, default=str)}])
        except Exception as exc:
            return {"error": str(exc)}
        text = ((resp["choices"][0].get("message") or {}).get("content") or "").translate(_ASCII)
        parsed = _parse_json(text)
        out = parsed if parsed else {"raw": text.strip()[:1500]}
        out["model"] = resp.get("model", self.client.full_model_str)
        return out

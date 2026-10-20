"""
src/agentic/tools.py
====================
Tool schemas (OpenAI / LiteLLM function-calling format) and their Python
implementations for the LLM agents:

  4B Adversarial Strategist — get_round_metrics, get_shap_drift,
                              get_enforcement_notes, propose_mutation
  4C Analyst Assistant      — list_rounds, query_flags, query_round_metrics,
                              get_trader_history

Every tool reads only what the pipeline has written to disk for the current
run. When the requested data does not exist a tool returns {"error": ...} —
it never substitutes made-up numbers (SRS FR-4.3, §9 "run/round not found").

propose_mutation is the only tool with a side effect. Proposals are validated
against PARAM_BOUNDS (persona allow-list, known parameter names, types and
ranges) before anything is written to configs/persona_config.yaml (FR-4.2).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.core.agents.manipulation.common import BAND as NEAR_TOUCH_BAND
from src.detector.pipeline import read_flags

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
TACTIC_CARDS_PATH = REPO_ROOT / "docs" / "tactic_cards.yaml"
PERSONA_CONFIG_PATH = REPO_ROOT / "configs" / "persona_config.yaml"
ENFORCEMENT_NOTES_PATH = REPO_ROOT / "docs" / "enforcement_case_notes.md"

# =============================================================================
# Hard parameter bounds for strategist proposals (single source of truth: the
# strategist prompt is generated from this table too).
# =============================================================================

PARAM_BOUNDS: dict[str, dict[str, tuple[type, float, float, str]]] = {
    "spoofing": {
        "order_size_multiplier": (float, 1.0, 15.0, "spoof size vs. average visible level depth"),
        "cancellation_delay_ticks": (int, 1, 30, "mean ticks the spoof rests before cancel"),
        "cancel_jitter": (float, 0.0, 0.8, "cancel delay drawn from delay x U(1-j, 1+j)"),
        "size_jitter": (float, 0.0, 0.8, "spoof size x U(1-j, 1+j)"),
        "n_split_orders": (int, 1, 4, "spoof split into k orders on consecutive ticks"),
        "n_accounts": (int, 1, 3, ">=2: genuine and spoof legs from different accounts"),
        "frequency": (float, 0.05, 1.0, "chance per tick to start a new spoof cycle"),
        "price_aggressiveness": (float, 0.0001, 0.01, "distance behind best price; >0.002 loses impact"),
        "reprice_to_touch": (int, 0, 1, "1 = keep the resting spoof pinned behind the moving best price (Sarao)"),
        "flip_after_cancel": (float, 0.0, 1.0, "chance of trading aggressively in the genuine direction when the spoof is pulled (Oystacher)"),
        "direction_persistence": (float, 0.5, 0.99, "chance each new cycle keeps the previous spoof side; high = minutes-long one-sided campaigns (Sarao, Coscia), low = alternating sides whose pushes cancel each other"),
        "busy_period_bias": (float, 0.0, 1.0, "skip quiet periods; act mainly when the market is busy (design assumption)"),
        "cover_rate": (float, 0.0, 0.5, "per-account chance per tick of a genuine cover order inside the window (Sarao/Coscia traded genuinely too)"),
        "cover_size": (float, 0.0, 3.0, "cover order size x average level depth; the detector's ratios are volume-weighted, so small cover barely dilutes"),
    },
    "wash_trading": {
        "trade_frequency": (float, 0.05, 1.0, "total cross rate per tick (split across groups)"),
        "n_groups": (int, 1, 5, "independent colluding groups (dilutes, same volume)"),
        "ring_size": (int, 2, 6, "accounts per group; crosses rotate A->B->C->A"),
        "size_jitter": (float, 0.0, 0.9, "0 = fixed scripted lot; higher = varied sizes"),
        "camouflage_rate": (float, 0.0, 0.5, "share of cycles replaced by a genuine market trade"),
        "busy_period_bias": (float, 0.0, 1.0, "skip quiet periods; act mainly when the market is busy (design assumption)"),
    },
    "pump_and_dump": {
        "n_coordinated_accounts": (int, 2, 15, "colluding accounts"),
        "accumulation_size": (int, 50, 1000, "target position per account"),
        "burst_duration_ticks": (int, 50, 1000, "accumulate + ignite duration (ignite = last 30%)"),
        "dump_delay_ticks": (int, 10, 500, "passive distribution period before the dump"),
        "stagger": (float, 0.0, 1.0, "per-account random start delay, x burst duration"),
        "ignition_aggression": (float, 0.0002, 0.005, "how far through the ask ignition buys go"),
        "decoy_rate": (float, 0.0, 0.4, "chance per tick of a small opposite-side trade"),
        "busy_period_bias": (float, 0.0, 1.0, "skip quiet periods; act mainly when the market is busy (design assumption)"),
    },
    "layering": {
        "n_layers": (int, 2, 8, "orders stacked on consecutive levels per cycle (Tower: 3-8)"),
        "layer_spacing": (float, 0.00005, 0.003, "gap between layers as a fraction of price (min one tick)"),
        "layer_size_multiplier": (float, 0.5, 6.0, "layer size vs. average visible level depth"),
        "size_taper": (float, 0.0, 0.9, "0 = equal layers; higher = smaller layers deeper in the book"),
        "cancellation_delay_ticks": (int, 1, 30, "mean ticks the layers rest before being cancelled"),
        "cancel_jitter": (float, 0.0, 0.8, "cancel delay drawn from delay x U(1-j, 1+j)"),
        "frequency": (float, 0.05, 1.0, "chance per tick to start a new layering cycle"),
        "busy_period_bias": (float, 0.0, 1.0, "skip quiet periods; act mainly when the market is busy (design assumption)"),
        "placement_stagger": (int, 0, 5, "ticks between successive layers: 0 = whole ladder in one tick; k = one layer every k ticks (no same-tick multi-level batch, but the full ladder rests for less of the cycle)"),
        "cover_rate": (float, 0.0, 0.5, "per-account chance per tick of a genuine cover order inside the window (Sarao/Coscia traded genuinely too)"),
        "cover_size": (float, 0.0, 3.0, "cover order size x average level depth; the detector's ratios are volume-weighted, so small cover barely dilutes"),
    },
}
MUTABLE_PERSONAS = tuple(PARAM_BOUNDS)

# Direction in which a knob LOWERS the persona's primary effect metric
# (effectiveness.PRIMARY_METRIC). Only knobs whose effect cost follows from how
# the metric is measured are listed:
#   fake_depth_share   spoof/layer volume inside the +-0.2% near-touch band
#   wash_volume_share  self-crossed share of market volume
#   ignition_pressure  peak 60-tick share of market volume bought aggressively
# Used by the effect guard (at risk: undo these; surplus: spend them on evasion).
EFFECT_COST: dict[str, dict[str, str]] = {
    "spoofing": {"order_size_multiplier": "down", "price_aggressiveness": "up", "direction_persistence": "down"},
    "layering": {"layer_spacing": "up", "n_layers": "down", "layer_size_multiplier": "down", "size_taper": "up",
                 "placement_stagger": "up"},
    "wash_trading": {"trade_frequency": "down", "camouflage_rate": "up"},
    "pump_and_dump": {"ignition_aggression": "down", "accumulation_size": "down",
                      "n_coordinated_accounts": "down", "burst_duration_ticks": "up"},
}

# Values past which an effect-costing knob takes the scheme outside what its effect
# metric measures, so the effect collapses instead of fading: a spoof resting further
# than the near-touch band behind the best price adds no near-touch fake depth. Code
# exploration stays short of these (every probed price_aggressiveness jump from 0.001
# landed at 0.0036-0.0086 and scored effect 0.00 in rounds 1-3 of run 113045Z_seed314).
EFFECT_CLIFF: dict[str, dict[str, float]] = {
    "spoofing": {"price_aggressiveness": NEAR_TOUCH_BAND},
}


def _costs_effect(persona: str, knob: str, old, new) -> bool:
    """True when old -> new moves `knob` in the direction that lowers the effect."""
    d = EFFECT_COST.get(persona, {}).get(knob)
    if d is None or old is None or new is None or old == new:
        return False
    return (new > old) if d == "up" else (new < old)


def validate_mutation(persona: str, parameters: Any) -> tuple[dict[str, Any] | None, str | None]:
    """Returns (clean_parameters, None) or (None, error_message)."""
    if persona not in PARAM_BOUNDS:
        return None, f"Unknown or non-mutable persona '{persona}'. Allowed: {list(MUTABLE_PERSONAS)}."
    if not isinstance(parameters, dict) or not parameters:
        return None, "parameters must be a non-empty object of {name: value}."
    bounds = PARAM_BOUNDS[persona]
    clean: dict[str, Any] = {}
    errors = []
    for name, value in parameters.items():
        if name not in bounds:
            errors.append(f"'{name}' is not a {persona} parameter (allowed: {list(bounds)})")
            continue
        typ, lo, hi, _ = bounds[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            errors.append(f"{name} must be a number, got {value!r}")
            continue
        if typ is int and float(value) != int(value):
            errors.append(f"{name} must be an integer, got {value}")
            continue
        v = typ(value)
        if not (lo <= v <= hi):
            errors.append(f"{name}={v} is outside [{lo}, {hi}]")
            continue
        clean[name] = v
    if errors:
        return None, "; ".join(errors)
    return clean, None


def bounds_prompt_text(defaults: dict[str, dict[str, Any]] | None = None) -> str:
    """Human-readable bounds table for the strategist's prompt."""
    lines = []
    for persona, params in PARAM_BOUNDS.items():
        if defaults is not None and persona not in defaults:
            continue
        lines.append(f" - {persona}:")
        for name, (typ, lo, hi, desc) in params.items():
            d = (defaults or {}).get(persona, {}).get(name)
            dflt = f"; now {d}" if d is not None else ""
            lines.append(f"   * {name} ({typ.__name__}, {lo}..{hi}{dflt}): {desc}")
    return "\n".join(lines)


# =============================================================================
# SCHEMAS
# =============================================================================

def _fn(name: str, description: str, properties: dict, required: list[str] | None = None) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": properties, "required": required or []},
        },
    }


GET_ROUND_METRICS_SCHEMA = _fn(
    "get_round_metrics",
    "Precision, recall, F1, FPR, event detection and latency per persona for a round "
    "(post-retrain, plus pre_retrain = frozen previous detector when available).",
    {"round_num": {"type": "integer", "description": "Round number (0 = baseline)"}},
    ["round_num"],
)
GET_SHAP_DRIFT_SCHEMA = _fn(
    "get_shap_drift",
    "Which features gained or lost mean |SHAP| importance in a persona's classifier between two rounds.",
    {
        "persona": {"type": "string"},
        "round_a": {"type": "integer"},
        "round_b": {"type": "integer"},
    },
    ["persona", "round_a", "round_b"],
)
GET_ENFORCEMENT_NOTES_SCHEMA = _fn(
    "get_enforcement_notes",
    "Qualitative CFTC enforcement case notes for a persona type.",
    {"persona": {"type": "string"}},
    ["persona"],
)
PROPOSE_MUTATION_SCHEMA = _fn(
    "propose_mutation",
    "Propose the next round's parameters for ONE persona. Validated against hard bounds; "
    "THIS IS THE ONLY TOOL WITH A SIDE EFFECT (writes configs/persona_config.yaml).",
    {
        "persona": {"type": "string", "enum": list(MUTABLE_PERSONAS)},
        "parameters": {"type": "object", "description": "Parameter dict, e.g. {'cancellation_delay_ticks': 10}"},
        "reasoning": {
            "type": "string",
            "description": "Step-by-step justification referencing the metrics, SHAP drift and case notes.",
        },
        "alternatives": {
            "type": "array", "maxItems": 2,
            "items": {"type": "object", "description": "A parameter dict, same shape as `parameters`"},
            "description": "Up to 2 alternative parameter dicts using DIFFERENT tactics. Every candidate is "
                           "probe-tested against the frozen detector and the best-scoring one is kept.",
        },
    },
    ["persona", "parameters", "reasoning"],
)
STRATEGIST_TOOLS = [GET_ROUND_METRICS_SCHEMA, GET_SHAP_DRIFT_SCHEMA, GET_ENFORCEMENT_NOTES_SCHEMA, PROPOSE_MUTATION_SCHEMA]

_MARKET = {"type": "string", "description": "Test market: 'A' (dir 'test') or 'B' (dir 'test_1'); omit for all test markets of the round"}
_PERSONA = {"type": "string", "description": "spoofing | layering | wash_trading | pump_and_dump | anomaly"}


_COMMODITY_ALIASES = {"wti": "crude_oil_wti", "crude": "crude_oil_wti", "crude oil": "crude_oil_wti", "oil": "crude_oil_wti",
                      "brent": "brent_crude", "gas": "natural_gas", "natgas": "natural_gas", "natural gas": "natural_gas"}


def _commodity_key(name: str) -> str:
    """Accept 'Wheat', 'natural gas', 'WTI', 'brent_crude' ... and return the data's commodity key."""
    k = str(name).strip().lower()
    return _COMMODITY_ALIASES.get(k, k.replace(" ", "_"))


def _account_status_counts(rows: list[dict]) -> dict:
    """
    Account-level ground-truth counts. "Account" = one trader flagged for one persona in one commodity.
      accounts_confirmed               at least one of its cases matches a real episode of that persona
      accounts_only_false_alerts       every case had no ground-truth episode
      accounts_with_any_false_alert    at least one case had no episode (overlaps the two above)
    """
    out = {"accounts_confirmed": 0, "accounts_other_persona_only": 0, "accounts_only_false_alerts": 0, "accounts_with_any_false_alert": 0}
    for a in rows:
        st = a["statuses"]
        if st.get("confirmed"):
            out["accounts_confirmed"] += 1
        elif st.get("other_persona"):
            out["accounts_other_persona_only"] += 1
        else:
            out["accounts_only_false_alerts"] += 1
        if st.get("no_episode"):
            out["accounts_with_any_false_alert"] += 1
    return out


def _read_alerted(path: Path) -> pd.DataFrame:
    """
    Alerted rows of flags.parquet with the SHAP / score structs cleaned. Filtering in Arrow
    before conversion avoids turning every window's struct into a dict (~13 s -> <1 s per round).
    """
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    from src.detector.pipeline import clean_shap_dict
    table = pq.read_table(path)
    df = table.filter(pc.equal(table["final_flag"], True)).to_pandas()
    df["top_shap_features"] = df["top_shap_features"].apply(clean_shap_dict)
    if "persona_scores" in df.columns:
        df["persona_scores"] = df["persona_scores"].apply(clean_shap_dict)
    df["triggered_by"] = df["triggered_by"].apply(lambda x: list(x) if x is not None else [])
    return df


def _r(v, nd: int = 4):
    return round(float(v), nd) if isinstance(v, (int, float)) and v == v else None


ANALYST_TOOLS = [
    _fn("list_rounds", "List the completed rounds of the current run with their flag counts.", {}),
    _fn(
        "summarize_cases",
        "BEST FOR 'which accounts were flagged and why'. Merges a round's alerted windows into cases per account, "
        "checks each against the simulator's ground truth (confirmed / other_persona / no_episode) and returns one row "
        "per account: windows, tick span, max score, top SHAP features (mean value; + pushes toward the flag) and totals.",
        {"round_num": {"type": "integer"}, "persona": _PERSONA, "market": _MARKET,
         "commodity": {"type": "string", "description": "Only this commodity, e.g. wheat"},
         "status": {"type": "string", "description": "confirmed | other_persona | no_episode"},
         "top_n": {"type": "integer", "description": "Accounts to return (default 10, max 25)"}},
        ["round_num"],
    ),
    _fn(
        "compare_markets",
        "BEST FOR comparing test markets (Market A vs B) or looking at one commodity's book in a round: per market and "
        "commodity the open/close, change %, range, 1-minute volatility, volume, trades, aggressive-buy share, ground-truth "
        "episodes by persona and the detector's alerts (accounts flagged, confirmed, false alerts).",
        {"round_num": {"type": "integer"},
         "commodity": {"type": "string", "description": "e.g. wheat, corn, gold, silver, copper, natural_gas, crude_oil_wti, brent_crude; omit for all"}},
        ["round_num"],
    ),
    _fn(
        "run_overview",
        "BEST FOR comparisons across personas or rounds (hardest persona, trends, evasion vs recovery). One call returns "
        "every round: retrained F1/recall/precision, frozen-detector coverage, effect vs round 0, mutation score and source.",
        {"market": _MARKET},
    ),
    _fn(
        "get_strategist_history",
        "What the adversarial LLM strategist changed for each persona each round, its reasoning and the outcome.",
        {"persona": _PERSONA},
    ),
    _fn(
        "query_flags",
        "Raw alerted WINDOWS (not accounts) for a round, filterable by trader_id / persona / market. Use for window-level "
        "detail on one account; prefer summarize_cases for lists of accounts.",
        {
            "round_num": {"type": "integer", "description": "Round number; omit for all rounds"},
            "trader_id": {"type": "string"},
            "persona": _PERSONA,
            "market": _MARKET,
            "limit": {"type": "integer", "description": "Max rows to return (default 10, max 25)"},
        },
    ),
    _fn(
        "query_round_metrics",
        "Full per-persona metrics for one round (pooled over test markets): precision/recall/F1/FPR, event detection, "
        "latency, coverage, for the frozen (pre) and retrained (post) detector.",
        {"round_num": {"type": "integer"}},
        ["round_num"],
    ),
    _fn(
        "get_trader_history",
        "One trader's orders and trades in a round (optionally one market and a tick range): volumes, cancels, "
        "largest order, counterparties.",
        {
            "trader_id": {"type": "string"},
            "round_num": {"type": "integer"},
            "market": _MARKET,
            "start_tick": {"type": "integer"},
            "end_tick": {"type": "integer"},
        },
        ["trader_id", "round_num"],
    ),
]


# =============================================================================
# IMPLEMENTATIONS
# =============================================================================


class ToolRegistry:
    """Python implementations of the tools, scoped to one run directory."""

    def __init__(self, run_dir=None, persona_config_path: Path | None = None):
        self.run_dir = Path(run_dir) if run_dir else None
        self.persona_config_path = Path(persona_config_path) if persona_config_path else PERSONA_CONFIG_PATH
        # persona -> [primary, *alternatives] parameter changes proposed this round (for probing)
        self.candidates: dict[str, list[dict]] = {}

    # ------------------------------------------------------------------ helpers
    def _round_dir(self, round_num: int) -> Path | None:
        if self.run_dir is None:
            return None
        d = self.run_dir / f"round_{int(round_num)}"
        return d if d.is_dir() else None

    def _metrics_json(self, round_num: int) -> dict | None:
        d = self._round_dir(round_num)
        if d is None or not (d / "metrics.json").exists():
            return None
        return json.loads((d / "metrics.json").read_text())

    def _available_rounds(self) -> list[int]:
        if self.run_dir is None or not self.run_dir.exists():
            return []
        return sorted(int(p.name.split("_")[1]) for p in self.run_dir.glob("round_*") if (p / "metrics.json").exists())

    def _not_found(self, what: str) -> dict:
        return {"error": f"{what} not found in this run.", "available_rounds": self._available_rounds()}

    # ------------------------------------------------------------- strategist
    def get_round_metrics(self, round_num: int) -> dict:
        """Post-retrain per-persona metrics for a round, or {"error": ...}."""
        data = self._metrics_json(round_num)
        if data is None:
            return self._not_found(f"Metrics for round {round_num}")
        return data.get("personas", {})

    def get_round_metrics_full(self, round_num: int) -> dict:
        data = self._metrics_json(round_num)
        if data is None:
            return self._not_found(f"Metrics for round {round_num}")
        return {
            "post_retrain": data.get("personas", {}),
            "pre_retrain": data.get("pre_retrain"),
            "unsupervised_isolation_forest": data.get("unsupervised"),
        }

    def _shap_importance(self, round_num: int) -> dict:
        d = self._round_dir(round_num)
        if d is None:
            return {}
        for name in ("shap_importance.json", "feature_importances.json"):
            if (d / name).exists():
                return json.loads((d / name).read_text())
        return {}

    def get_shap_drift(self, persona: str, round_a: int, round_b: int) -> dict:
        """Mean |SHAP| change per feature for `persona` between two rounds, plus metric deltas."""
        m_a = self.get_round_metrics(round_a).get(persona) if "error" not in self.get_round_metrics(round_a) else None
        m_b = self.get_round_metrics(round_b).get(persona) if "error" not in self.get_round_metrics(round_b) else None
        if not m_a or not m_b:
            return {"persona": persona, "error": f"Missing metrics for round {round_a} or {round_b}."}

        def delta(k):
            a, b = m_a.get(k), m_b.get(k)
            return None if a is None or b is None else round(b - a, 4)

        result: dict[str, Any] = {
            "persona": persona, "round_a": round_a, "round_b": round_b,
            "f1_delta": delta("f1"), "precision_delta": delta("precision"), "recall_delta": delta("recall"),
            "interpretation": "Negative delta = detector got worse (evasion worked); positive = it recovered.",
        }
        si_a = self._shap_importance(round_a).get(persona, {})
        si_b = self._shap_importance(round_b).get(persona, {})
        if si_a and si_b:
            deltas = {f: round(si_b.get(f, 0.0) - si_a.get(f, 0.0), 5) for f in set(si_a) | set(si_b)}
            deltas = {k: v for k, v in sorted(deltas.items(), key=lambda kv: abs(kv[1]), reverse=True) if abs(v) > 1e-4}
            result["shap_importance_round_b_top"] = dict(list(si_b.items())[:8])
            result["shap_importance_deltas"] = dict(list(deltas.items())[:12])
            result["top_rising_features"] = [k for k, v in deltas.items() if v > 0][:5]
            result["top_declining_features"] = [k for k, v in deltas.items() if v < 0][:5]
        else:
            result["note"] = "SHAP importance files not found for both rounds; metric deltas only."
        return result

    def get_track_record(self, persona: str, max_rounds: int = 8) -> list[dict]:
        """
        What every past round did for `persona` and how it went: parameters in
        force, who chose them, recall of the frozen (pre) and retrained (post)
        detector, effectiveness vs. round 0 and the resulting mutation score.
        """
        rows = []
        for r in self._available_rounds()[-max_rounds:]:
            m = self._metrics_json(r) or {}
            pre = (m.get("pre_retrain") or {}).get(persona, {})
            post = (m.get("personas") or {}).get(persona, {})
            eff = (m.get("effectiveness") or {}).get(persona, {})
            key = eff.get("primary_metric")
            from src.adversarial.effectiveness import evasion_components
            rows.append({
                "round": r,
                "source": (m.get("mutation_source") or {}).get(persona),
                "params": (m.get("params_used") or {}).get(persona, {}),
                "frozen_recall": pre.get("recall"),
                "frozen_f1": pre.get("f1"),
                "frozen_coverage": pre.get("caught_coverage", pre.get("episode_coverage")),
                "own_label_coverage": pre.get("episode_coverage"),
                "episodes": (f"{pre.get('caught_events_detected', pre.get('events_detected'))}/{pre.get('events_total')}"
                             if pre else None),
                "frozen_delay_ticks": pre.get("caught_latency_ticks", pre.get("detection_latency_ticks")),
                "evasion_parts": evasion_components(pre) if pre else None,
                "retrained_recall": post.get("recall"),
                "effect": {key: eff.get(key)} if key else {},
                "effect_ratio": (m.get("effectiveness_ratio") or {}).get(persona),
                "score": (m.get("mutation_score") or {}).get(persona),
            })
        return rows

    def effect_status(self, persona: str) -> dict:
        """
        Is the persona's last evaluated parameter set still manipulating?
          status  "failed"  effect ratio < EFFECTIVENESS_FLOOR (score is 0)
                  "at_risk" effect ratio < EFFECT_AT_RISK
                  "surplus" effect ratio >= EFFECT_SURPLUS (unused effect budget)
                  "ok"      otherwise (or no ratio yet)
          anchor  failed: the best effective past round (highest score among rounds
                  whose effect ratio cleared the floor; round 0 counts as
                  effective, ties go to the most recent) — the reset target.
                  at_risk: the latest earlier round with effect >= EFFECT_AT_RISK.
                  Returned with its params and the knob changes from it to the
                  last parameters.
        """
        from src.adversarial.effectiveness import EFFECT_AT_RISK, EFFECT_SURPLUS, EFFECTIVENESS_FLOOR
        rows = [r for r in self.get_track_record(persona, max_rounds=50) if r["params"]]
        if not rows:
            return {"status": "ok"}
        last = rows[-1]
        ratio = last.get("effect_ratio")
        status = "ok"
        if ratio is not None and ratio < EFFECTIVENESS_FLOOR:
            status = "failed"
        elif ratio is not None and ratio < EFFECT_AT_RISK:
            status = "at_risk"
        elif ratio is not None and ratio >= EFFECT_SURPLUS:
            status = "surplus"
        out = {"status": status, "round": last["round"], "effect_ratio": ratio, "params": last["params"]}
        if status == "surplus":
            return out
        if status == "failed":   # reset target: best score among sets that still manipulated
            effective = [r for r in rows if r["round"] == 0 or (r.get("effect_ratio") or 0) >= EFFECTIVENESS_FLOOR]
            pick = lambda c: max(c, key=lambda r: ((r.get("score") or 0.0), r["round"]))  # noqa: E731
        else:                    # at risk: the latest set that was comfortably effective
            effective = [r for r in rows[:-1] if r["round"] == 0 or (r.get("effect_ratio") or 0) >= EFFECT_AT_RISK]
            pick = lambda c: max(c, key=lambda r: r["round"])  # noqa: E731
        if effective and status != "ok":
            anchor = pick(effective)
            out.update({
                "anchor_round": anchor["round"],
                "anchor_effect_ratio": anchor.get("effect_ratio") if anchor["round"] else 1.0,
                "anchor_params": anchor["params"],
                "changes_since_anchor": {k: [anchor["params"].get(k), v] for k, v in last["params"].items()
                                         if anchor["params"].get(k) != v},
            })
        return out

    def enforce_effect_guard(self, persona: str, status: dict) -> dict[str, list]:
        """
        Code-enforced at-risk rule, applied AFTER this round's mutation for a
        persona whose effect was at risk (see effect_status):
          1. any NEW change that lowers the effect (EFFECT_COST, versus the last
             round's parameters) is undone;
          2. if the proposal undid none of the effect-costing changes made since
             the anchor round, all of them are reverted to the anchor values
             (evasion-only knobs such as cancel timing are left as proposed).
        Returns {knob: [value_before_guard, value_after]} (empty when nothing changed).
        """
        if status.get("status") != "at_risk":
            return {}
        cfg = self._load_persona_config()
        cur = dict(cfg.get(persona) or {})
        last, anchor = status.get("params") or {}, status.get("anchor_params") or {}
        reverts: dict[str, list] = {}
        for k, v in list(cur.items()):
            if _costs_effect(persona, k, last.get(k), v):
                reverts[k] = [v, last[k]]
                cur[k] = last[k]
        harmful = {k: (anchor[k], last[k]) for k in EFFECT_COST.get(persona, {})
                   if k in anchor and k in last and _costs_effect(persona, k, anchor[k], last[k])}
        undone = any(cur.get(k) != was for k, (_a, was) in harmful.items())
        if harmful and not undone:
            for k in harmful:
                reverts[k] = [cur.get(k), anchor[k]]
                cur[k] = anchor[k]
        if reverts:
            res = self.propose_mutation(persona, {k: v for k, (_old, v) in reverts.items()}, reasoning="effect guard")
            if "error" in res:
                logger.warning(f"[Tools] effect guard for {persona} failed: {res['error']}")
                return {}
            logger.info(f"[Tools] {persona}: effect at risk, guard reverted {reverts}")
        return reverts

    def effect_costing_knobs(self, persona: str) -> str:
        """Prompt text: which knob directions lower this persona's effect."""
        return ", ".join(f"{k} {d}" for k, d in EFFECT_COST.get(persona, {}).items())

    def recover_effect(self, persona: str) -> dict | None:
        """
        Code-enforced recovery: when the last round's effect fell below the floor,
        reset the persona's parameters to its best effective past set (see
        effect_status) BEFORE the next mutation is applied on top. Returns the
        status dict when a reset happened, else None.
        """
        st = self.effect_status(persona)
        if st["status"] != "failed" or "anchor_params" not in st:
            return None
        res = self.propose_mutation(persona, st["anchor_params"], reasoning="effect recovery")
        if "error" in res:
            logger.warning(f"[Tools] effect recovery for {persona} failed: {res['error']}")
            return None
        logger.info(f"[Tools] {persona}: effect {st['effect_ratio']}x < floor in round {st['round']}; "
                    f"parameters reset to round {st['anchor_round']}")
        return st

    def knob_usage(self, persona: str) -> dict[str, list[str]]:
        """Knobs never changed from round 0, and knobs currently at a bound (exploration hints)."""
        history = [row["params"] for row in self.get_track_record(persona, max_rounds=50) if row["params"]]
        bounds = PARAM_BOUNDS.get(persona, {})
        if not history:
            return {"never_changed": list(bounds), "at_bound": []}
        first, last = history[0], history[-1]
        never = [k for k in bounds if all(h.get(k) == first.get(k) for h in history)]
        at_bound = []
        for k, (_typ, lo, hi, _d) in bounds.items():
            v = last.get(k)
            if isinstance(v, (int, float)) and (v <= lo or v >= hi):
                at_bound.append(f"{k}={v} ({'min' if v <= lo else 'max'})")
        return {"never_changed": never, "at_bound": at_bound}

    def get_tactic_cards(self, persona: str) -> list[str]:
        """Compact one-line tactic cards (docs/tactic_cards.yaml)."""
        import yaml
        if not TACTIC_CARDS_PATH.exists():
            return []
        cards = (yaml.safe_load(TACTIC_CARDS_PATH.read_text(encoding="utf-8")) or {}).get(persona, [])
        return [
            f"{c['tactic']} [{c['source']}]: knobs {c['knobs']}; weakens {', '.join(c['weakens'])}; cost: {c['cost']}"
            for c in cards
        ]

    def get_enforcement_notes(self, persona: str) -> str:
        """Relevant section of docs/enforcement_case_notes.md for a persona."""
        if ENFORCEMENT_NOTES_PATH.exists():
            content = ENFORCEMENT_NOTES_PATH.read_text(encoding="utf-8")
            keyword = persona.lower().replace("_", " ")
            for sec in content.split("\n# "):
                header = sec.split("\n", 1)[0].lower()
                if keyword in header or persona.replace("_", "-") in header:
                    lines = [line for line in sec.splitlines() if line.strip()]
                    return "# " + "\n".join(lines[:25])
        return f"No enforcement notes found for persona '{persona}'."

    def _load_persona_config(self) -> dict:
        import yaml
        if self.persona_config_path.exists():
            with open(self.persona_config_path) as f:
                return yaml.safe_load(f) or {}
        return {}

    def propose_mutation(self, persona: str, parameters: dict, reasoning: str = "",
                         alternatives: list | None = None) -> dict:
        """Validate against PARAM_BOUNDS, then write under the top-level persona key.
        Valid `alternatives` are kept in self.candidates for probe testing (invalid ones are dropped)."""
        import yaml
        clean, error = validate_mutation(persona, parameters)
        if error:
            return {"error": f"Validation failed: {error}"}
        alts, dropped = [], []
        for alt in (alternatives or [])[:2]:
            # models sometimes nest an alternative as {"parameters": {...}, "reasoning": ...}
            if isinstance(alt, dict) and isinstance(alt.get("parameters"), dict):
                alt = alt["parameters"]
            a, err = validate_mutation(persona, alt)
            (alts.append(a) if a else dropped.append(err))
        self.candidates[persona] = [clean, *alts]
        existing = self._load_persona_config()
        old_params = dict(existing.get(persona) or existing.get("round_0", {}).get(persona, {}))
        merged = dict(old_params)
        merged.update(clean)
        existing[persona] = {k: v for k, v in merged.items() if k in PARAM_BOUNDS[persona]}
        with open(self.persona_config_path, "w") as f:
            yaml.dump(existing, f, default_flow_style=False)
        return {
            "status": "success",
            "message": f"Mutated {persona} config.",
            "old_params": old_params,
            "new_params": existing[persona],
            "reasoning": reasoning,
            "alternatives_accepted": len(alts),
            **({"alternatives_dropped": dropped} if dropped else {}),
        }

    def caught_by(self, persona: str, round_num: int, top_n: int = 5) -> dict | None:
        """
        How the detector retrained at the end of `round_num` (the detector that will be
        FROZEN for the next round) treated this persona's episodes: share of episode
        windows flagged, the label mix of those flags and the features (sum of |SHAP| in
        the stored top attributions) that drove them.
        """
        d = self._round_dir(round_num)
        if d is None or not (d / "flags.parquet").exists():
            return None
        ev = [pd.read_parquet(m / "manipulation_events.parquet") for m in sorted(d.glob("test*"))
              if (m / "manipulation_events.parquet").exists()]
        if not ev:
            return None
        ev = pd.concat(ev, ignore_index=True)
        ev = ev[ev["persona"] == persona]
        if ev.empty:
            return None
        flags = read_flags(d / "flags.parquet")
        hit = pd.Series(False, index=flags.index)
        for _, e in ev.iterrows():
            hit |= ((flags["commodity"] == e["commodity"]) & flags["trader_id"].isin(set(e["trader_ids"]))
                    & (flags["window_end"] >= e["start_tick"]) & (flags["window_start"] <= e["end_tick"]))
        rows = flags[hit]
        caught = rows[rows["final_flag"].astype(bool)]
        if rows.empty:
            return None
        mix = caught["predicted_persona"].fillna("none").map(
            lambda v: "own label" if v == persona else ("anomaly" if v == "anomaly" else "other persona")).value_counts()
        weight: dict[str, float] = {}
        for d_ in caught["top_shap_features"]:
            for k, v in (d_ or {}).items():
                if v is not None:
                    weight[k] = weight.get(k, 0.0) + abs(float(v))
        total = sum(weight.values()) or 1.0
        top = dict(sorted(((k, round(v / total, 3)) for k, v in weight.items()), key=lambda kv: -kv[1])[:top_n])
        return {"windows": int(len(rows)), "caught_share": round(len(caught) / len(rows), 3),
                "label_mix": {k: round(v / max(1, len(caught)), 2) for k, v in mix.items()}, "top_features": top}

    # ---------------------------------------------------------------- analyst
    def list_rounds(self) -> dict:
        rounds = []
        for r in self._available_rounds():
            m = self._metrics_json(r) or {}
            rounds.append({"round_num": r, "n_flags": m.get("n_flags")})
        if not rounds:
            return {"error": "No completed rounds in this run."}
        return {"rounds": rounds}

    def _market_dirs(self, d: Path, market: str | None = None) -> list[Path]:
        """A round's test markets ('test', 'test_1', ...); `market` may be the dir name or 'A'/'B'/'Market A'."""
        dirs = [x for x in [d / "test"] + sorted(d.glob("test_*")) if x.is_dir()]
        if market is None or market == "":
            return dirs
        m = str(market).strip()
        if m.lower().startswith("market "):
            m = m.split()[-1]
        if len(m) == 1 and m.isalpha():
            idx = ord(m.upper()) - ord("A")
            return dirs[idx: idx + 1]
        return [x for x in dirs if x.name == m]

    def _tick_range(self, market_dir: Path) -> tuple[int, int] | None:
        p = market_dir / "trade_log.parquet"
        if not p.exists():
            return None
        t = pd.read_parquet(p, columns=["tick"])["tick"]
        return (int(t.min()), int(t.max())) if len(t) else None

    def _flagged(self, round_num: int, market: str | None = None) -> pd.DataFrame | None:
        """Alerted windows of a round, optionally only those inside one test market's ticks."""
        d = self._round_dir(round_num)
        if d is None or not (d / "flags.parquet").exists():
            return None
        df = _read_alerted(d / "flags.parquet")
        if market:
            dirs = self._market_dirs(d, market)
            if not dirs:
                return df.iloc[0:0]
            rng = self._tick_range(dirs[0])
            if rng:
                df = df[(df["window_start"] >= rng[0]) & (df["window_start"] <= rng[1])]
        return df

    def query_flags(self, round_num: int | None = None, trader_id: str | None = None,
                    persona: str | None = None, limit: int = 10, market: str | None = None) -> dict:
        rounds = [round_num] if round_num is not None else self._available_rounds()
        frames = []
        for r in rounds:
            df = self._flagged(r, market)
            if df is None:
                if round_num is not None:
                    return self._not_found(f"Flags for round {round_num}")
                continue
            df = df.copy()
            df.insert(0, "round_num", r)
            frames.append(df)
        if not frames:
            return self._not_found("Flags")
        df = pd.concat(frames, ignore_index=True)
        if trader_id:
            df = df[df["trader_id"] == trader_id]
        if persona:
            df = df[df["predicted_persona"] == persona]
        df = df.sort_values("final_score", ascending=False)
        rows = []
        for _, r in df.head(max(1, min(int(limit or 10), 25))).iterrows():
            rows.append({
                "round_num": int(r["round_num"]),
                "commodity": r["commodity"],
                "trader_id": r["trader_id"],
                "window_start": int(r["window_start"]),
                "window_end": int(r["window_end"]),
                "predicted_persona": r["predicted_persona"],
                "final_score": round(float(r["final_score"]), 4),
                "triggered_by": list(r["triggered_by"]),
                "top_shap_features": r["top_shap_features"],
            })
        return {"total_matching_flags": int(len(df)), "returned": len(rows), "flags": rows}

    def query_round_metrics(self, round_num: int) -> dict:
        return self.get_round_metrics_full(round_num)

    def summarize_cases(self, round_num: int, persona: str | None = None, market: str | None = None,
                        status: str | None = None, top_n: int = 10, commodity: str | None = None,
                        _max_rows: int = 25) -> dict:
        """
        Account-level view of a round's alerts: consecutive flagged windows of one account and
        persona are merged into a case, matched to the simulator's ground truth, then grouped
        per account. This is the right granularity for "which accounts and why".
        """
        df = self._flagged(round_num, market)
        if df is None:
            return self._not_found(f"Flags for round {round_num}")
        if persona:
            df = df[df["predicted_persona"] == persona]
        if commodity:
            df = df[df["commodity"] == _commodity_key(commodity)]
        d = self._round_dir(round_num)
        ev = [pd.read_parquet(x / "manipulation_events.parquet") for x in self._market_dirs(d, market)
              if (x / "manipulation_events.parquet").exists()]
        ev = pd.concat(ev, ignore_index=True) if ev else pd.DataFrame(columns=["event_id", "persona", "commodity", "trader_ids", "start_tick", "end_tick"])
        if commodity:
            ev = ev[ev["commodity"] == _commodity_key(commodity)]

        cases = []
        for (comm, trader, pers), g in df.sort_values("window_start").groupby(["commodity", "trader_id", "predicted_persona"], sort=False):
            cur = None
            for r in g.itertuples(index=False):
                if cur and r.window_start <= cur["end_tick"] + 1:
                    cur["end_tick"] = max(cur["end_tick"], int(r.window_end)); cur["windows"] += 1
                    cur["max_score"] = max(cur["max_score"], float(r.final_score)); cur["shap"].append(r.top_shap_features or {})
                    continue
                if cur:
                    cases.append(cur)
                cur = {"commodity": comm, "trader_id": trader, "persona": pers, "start_tick": int(r.window_start),
                       "end_tick": int(r.window_end), "windows": 1, "max_score": float(r.final_score), "shap": [r.top_shap_features or {}]}
            if cur:
                cases.append(cur)

        matched = set()
        for c in cases:
            c["status"], c["ground_truth"] = "no_episode", None
            for e in ev.itertuples(index=False):
                if e.commodity == c["commodity"] and c["trader_id"] in set(e.trader_ids) and not (c["end_tick"] < e.start_tick or c["start_tick"] > e.end_tick):
                    c["ground_truth"] = e.persona
                    c["status"] = "confirmed" if e.persona == c["persona"] else "other_persona"
                    if e.persona == c["persona"]:
                        matched.add(e.event_id)
                    break
        # Accounts are built from ALL their cases; a status filter then keeps accounts that have at
        # least one case with that status (filtering cases first made every account look single-status).
        accounts: dict[tuple, dict] = {}
        for c in cases:
            a = accounts.setdefault((c["trader_id"], c["commodity"], c["persona"]), {
                "trader_id": c["trader_id"], "commodity": c["commodity"], "persona": c["persona"], "cases": 0, "windows": 0,
                "first_tick": c["start_tick"], "last_tick": c["end_tick"], "max_score": 0.0, "statuses": {}, "_w": {}})
            a["cases"] += 1; a["windows"] += c["windows"]; a["max_score"] = max(a["max_score"], c["max_score"])
            a["first_tick"] = min(a["first_tick"], c["start_tick"]); a["last_tick"] = max(a["last_tick"], c["end_tick"])
            a["statuses"][c["status"]] = a["statuses"].get(c["status"], 0) + 1
            if c["ground_truth"] and c["ground_truth"] != c["persona"]:
                a["ground_truth_persona"] = c["ground_truth"]
            for s in c["shap"]:
                for k, v in s.items():
                    if v is not None:
                        a["_w"].setdefault(k, []).append(float(v))
        rows = []
        for a in accounts.values():
            w = a.pop("_w")
            top = sorted(w.items(), key=lambda kv: -sum(abs(x) for x in kv[1]))[:3]
            a["top_features"] = {k: round(sum(v) / len(v), 3) for k, v in top}   # mean SHAP; + pushes toward the flag
            a["max_score"] = round(a["max_score"], 3)
            rows.append(a)
        rows.sort(key=lambda a: (-a["max_score"], -a["windows"]))
        acct_status = _account_status_counts(rows)
        if status:
            rows = [a for a in rows if a["statuses"].get(status)]
            rows.sort(key=lambda a: (len(a["statuses"]) > 1, -a["max_score"]))   # accounts with ONLY that status first
        n_eps = int((ev["persona"] == persona).sum()) if persona else int(len(ev))
        return {
            "round_num": round_num, "market": market or "all test markets", "persona": persona or "all",
            **({"status_filter": f"accounts with at least one '{status}' case (their 'statuses' show all their cases)",
                "accounts_matching_filter": len(rows)} if status else {}),
            "totals": {"accounts": len(accounts), "accounts_by_status": acct_status,
                       "cases": len(cases), "flagged_windows": int(sum(c["windows"] for c in cases)),
                       "confirmed_cases": sum(c["status"] == "confirmed" for c in cases),
                       "other_persona_cases": sum(c["status"] == "other_persona" for c in cases),
                       "no_episode_cases": sum(c["status"] == "no_episode" for c in cases),
                       "ground_truth_episodes": n_eps,
                       "episodes_missed": n_eps - len({m for m in matched if not persona or m in set(ev[ev["persona"] == persona]["event_id"])})},
            "accounts_returned": min(len(rows), max(1, min(int(top_n or 10), _max_rows))),
            "accounts": rows[: max(1, min(int(top_n or 10), _max_rows))],
            "status_meaning": {"confirmed": "ground truth has this account, persona and span",
                               "other_persona": "a real episode, but of another persona",
                               "no_episode": "no ground-truth manipulation (false alert)"},
        }

    def compare_markets(self, round_num: int, commodity: str | None = None) -> dict:
        """
        The round's test markets side by side (each is an independent simulation), per commodity:
        price change, range, volatility, volume, trades, aggressive-buy share, ground-truth episodes
        by persona, and the detector's alerts (accounts flagged, confirmed, false alerts).
        """
        d = self._round_dir(round_num)
        if d is None:
            return self._not_found(f"Round {round_num}")
        dirs = self._market_dirs(d)
        if not dirs:
            return self._not_found(f"Round {round_num} test markets")
        comm = _commodity_key(commodity) if commodity else None
        out = {}
        for i, md in enumerate(dirs):
            label = chr(ord("A") + i)
            if not (md / "trade_log.parquet").exists():
                continue
            t = pd.read_parquet(md / "trade_log.parquet", columns=["tick", "price", "quantity", "commodity", "aggressor_side"])
            if comm:
                t = t[t["commodity"] == comm]
                if t.empty:
                    return {"error": f"Commodity '{commodity}' not found.",
                            "available_commodities": sorted(pd.read_parquet(md / "trade_log.parquet", columns=["commodity"])["commodity"].unique())}
            ev = pd.read_parquet(md / "manipulation_events.parquet") if (md / "manipulation_events.parquet").exists() else pd.DataFrame()
            per = {}
            for c, g in t.groupby("commodity"):
                g = g.sort_values("tick")
                px = g.groupby("tick")["price"].last()
                px = px.reindex(range(int(px.index.min()), int(px.index.max()) + 1)).ffill()
                r = np.log(px.iloc[::60]).diff().dropna() * 1e4
                e = ev[ev["commodity"] == c] if not ev.empty else ev
                first, last = float(g["price"].iloc[0]), float(g["price"].iloc[-1])
                per[c] = {
                    "open": _r(first), "close": _r(last), "change_pct": _r((last / first - 1) * 100, 3),
                    "range_bp": _r((g["price"].max() - g["price"].min()) / first * 1e4, 1),
                    "volatility_1min_bp": _r(float(r.std()), 2) if len(r) > 2 else None,
                    "volume": int(g["quantity"].sum()), "trades": int(len(g)),
                    "aggressive_buy_share": _r(float(g.loc[g["aggressor_side"] == "buy", "quantity"].sum() / max(1, g["quantity"].sum())), 3),
                    "episodes": {k: int(v) for k, v in e["persona"].value_counts().items()} if not e.empty else {},
                }
            # every account (internal call, not capped like the model-facing list)
            cases = self.summarize_cases(round_num, market=label, commodity=comm, top_n=10**6, _max_rows=10**6)
            by_comm: dict[str, list] = {}
            for a in cases.get("accounts", []):
                by_comm.setdefault(a["commodity"], []).append(a)
            for c in per:
                rows_c = by_comm.get(c, [])
                by_persona: dict[str, int] = {}
                for a in rows_c:
                    by_persona[a["persona"]] = by_persona.get(a["persona"], 0) + 1
                per[c]["alerts"] = {"accounts_flagged": len(rows_c), **_account_status_counts(rows_c), "by_persona": by_persona}
            out[f"Market {label}"] = {"dir": md.name, "commodities": per}
        return {"round_num": round_num, "commodity": comm or "all", "markets": out,
                "notes": "Markets are independent simulations of the same commodities; alerts come from the retrained detector "
                         "(alerts count every flagged account)."}

    def run_overview(self, market: str | None = None) -> dict:
        """
        Every round in one table: retrained F1/recall, frozen-detector coverage, effect vs round 0,
        mutation score and the mutation source per persona. With `market`, the retrained and effect
        figures come from that market when the run stored or derived them (by_market).
        """
        rounds = []
        for r in self._available_rounds():
            m = self._metrics_json(r) or {}
            scoped = None
            if market:
                d = self._round_dir(r)
                names = [x.name for x in self._market_dirs(d, market)] if d else []
                name = names[0] if names else None
                scoped = (m.get("by_market") or {}).get(name) if name else None
                if scoped is None and d is not None and name and (d / "by_market.json").exists():
                    scoped = json.loads((d / "by_market.json").read_text()).get(name)
            src = scoped or m
            per = {}
            for p in ("spoofing", "wash_trading", "pump_and_dump", "layering"):
                post = (src.get("personas") or {}).get(p) or {}
                pre = (src.get("pre_retrain") or (m.get("pre_retrain") if scoped else None) or {}).get(p) or {}
                per[p] = {"f1": _r(post.get("f1")), "recall": _r(post.get("recall")), "precision": _r(post.get("precision")),
                          "frozen_any_alert_coverage": _r(pre.get("caught_coverage", pre.get("episode_coverage"))),
                          "effect_ratio": _r((src.get("effectiveness_ratio") or {}).get(p)),
                          "mutation_score": _r((src.get("mutation_score") or (m.get("mutation_score") if scoped else None) or {}).get(p)),
                          "mutation_source": (m.get("mutation_source") or {}).get(p)}
            rounds.append({"round_num": r, "scope": (market if scoped else "pooled"), "personas": per})
        if not rounds:
            return {"error": "No completed rounds in this run."}
        # Precomputed per-persona conclusions: the model misreads rankings out of the nested round table.
        summary = {}
        for p in ("spoofing", "wash_trading", "pump_and_dump", "layering"):
            cov = [(r["round_num"], r["personas"][p]["frozen_any_alert_coverage"]) for r in rounds
                   if r["personas"][p]["frozen_any_alert_coverage"] is not None]
            f1 = [r["personas"][p]["f1"] for r in rounds if r["personas"][p]["f1"] is not None]
            worst = min(cov, key=lambda x: x[1]) if cov else (None, None)
            summary[p] = {"mean_frozen_coverage": _r(sum(c for _, c in cov) / len(cov)) if cov else None,
                          "lowest_frozen_coverage": worst[1], "lowest_frozen_coverage_round": worst[0],
                          "mean_retrained_f1": _r(sum(f1) / len(f1)) if f1 else None,
                          "latest_retrained_f1": f1[-1] if f1 else None}
        ranked = sorted((p for p in summary if summary[p]["mean_frozen_coverage"] is not None),
                        key=lambda p: summary[p]["mean_frozen_coverage"])
        return {"hardest_before_retraining_first": ranked,
                "persona_summary": summary,
                "rounds": rounds, "notes": "hardest_before_retraining_first ranks personas by mean frozen-detector coverage "
                "(lowest first = hardest to catch before retraining). "
                "frozen_any_alert_coverage = share of episode windows the detector frozen from the previous "
                "round caught (lower = better evasion); effect_ratio = manipulation effect vs round 0; frozen figures are pooled when "
                "the run did not store them per market."}

    def get_strategist_history(self, persona: str | None = None) -> dict:
        """What the strategist changed each round, its short reasoning and how the mutation did."""
        out = []
        for r in self._available_rounds():
            if r == 0:
                continue
            m = self._metrics_json(r) or {}
            d = self._round_dir(r)
            rec = json.loads((d / "strategist.json").read_text()).get("personas", {}) if d and (d / "strategist.json").exists() else {}
            for p, v in rec.items():
                if persona and p != persona:
                    continue
                pre = (m.get("pre_retrain") or {}).get(p) or {}
                out.append({"round_num": r, "persona": p, "source": v.get("source"),
                            "changes": {k: {"from": a, "to": b} for k, (a, b) in (v.get("changes") or {}).items()},
                            "reasoning": " ".join(str(v.get("reasoning") or "").split())[:400],
                            "frozen_any_alert_coverage": _r(pre.get("caught_coverage")),
                            "effect_ratio": _r((m.get("effectiveness_ratio") or {}).get(p)),
                            "mutation_score": _r((m.get("mutation_score") or {}).get(p))})
        if not out:
            return {"error": "No strategist rounds in this run (mutations start at round 1)."}
        return {"mutations": out}

    def get_trader_history(self, trader_id: str, round_num: int,
                           start_tick: int | None = None, end_tick: int | None = None, market: str | None = None) -> dict:
        d = self._round_dir(round_num)
        tests = [x for x in (self._market_dirs(d, market) if d else []) if (x / "order_log.parquet").exists()]
        if not tests:
            return self._not_found(f"Round {round_num} market logs")
        orders = pd.concat([pd.read_parquet(x / "order_log.parquet") for x in tests], ignore_index=True)
        trades = pd.concat([pd.read_parquet(x / "trade_log.parquet") for x in tests], ignore_index=True)
        orders = orders[orders["trader_id"] == trader_id]
        trades = trades[(trades["buyer_trader_id"] == trader_id) | (trades["seller_trader_id"] == trader_id)]
        if start_tick is not None:
            orders, trades = orders[orders["tick"] >= start_tick], trades[trades["tick"] >= start_tick]
        if end_tick is not None:
            orders, trades = orders[orders["tick"] <= end_tick], trades[trades["tick"] <= end_tick]
        if orders.empty and trades.empty:
            return {"error": f"No activity for trader '{trader_id}' in round {round_num} for the requested range."}
        placed = orders[orders["event_type"] == "placed"]
        cancelled = orders[orders["event_type"] == "cancelled"]
        bought = trades[trades["buyer_trader_id"] == trader_id]
        sold = trades[trades["seller_trader_id"] == trader_id]
        cpty = pd.concat([bought["seller_trader_id"], sold["buyer_trader_id"]])
        q = pd.concat([bought["quantity"], sold["quantity"]])
        cp_vol = q.groupby(cpty.to_numpy()).sum().sort_values(ascending=False)
        return {
            "trader_id": trader_id,
            "round_num": round_num,
            "first_tick": int(min(orders["tick"].min() if not orders.empty else 1e18,
                                  trades["tick"].min() if not trades.empty else 1e18)),
            "last_tick": int(max(orders["tick"].max() if not orders.empty else -1,
                                 trades["tick"].max() if not trades.empty else -1)),
            "orders_placed": int(len(placed)),
            "orders_placed_volume": int(placed["quantity"].sum()),
            "orders_cancelled": int(len(cancelled)),
            "orders_cancelled_volume": int(cancelled["quantity"].sum()),
            "largest_order": int(placed["quantity"].max()) if not placed.empty else 0,
            "trades": int(len(trades)),
            "bought_volume": int(bought["quantity"].sum()),
            "sold_volume": int(sold["quantity"].sum()),
            "net_position_change": int(bought["quantity"].sum() - sold["quantity"].sum()),
            "distinct_counterparties": int(cpty.nunique()),
            "top_counterparties_by_volume": {str(k): int(v) for k, v in cp_vol.head(3).items()},
        }

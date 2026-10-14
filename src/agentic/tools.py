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

import pandas as pd

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
        "busy_period_bias": (float, 0.0, 1.0, "skip quiet periods; act mainly when the market is busy (advisory #5)"),
        "cover_rate": (float, 0.0, 0.5, "per-account chance per tick of a genuine cover order inside the window (Sarao/Coscia traded genuinely too)"),
        "cover_size": (float, 0.0, 3.0, "cover order size x average level depth; the detector's ratios are volume-weighted, so small cover barely dilutes"),
    },
    "wash_trading": {
        "trade_frequency": (float, 0.05, 1.0, "total cross rate per tick (split across groups)"),
        "n_groups": (int, 1, 5, "independent colluding groups (dilutes, same volume)"),
        "ring_size": (int, 2, 6, "accounts per group; crosses rotate A->B->C->A"),
        "size_jitter": (float, 0.0, 0.9, "0 = fixed scripted lot; higher = varied sizes"),
        "camouflage_rate": (float, 0.0, 0.5, "share of cycles replaced by a genuine market trade"),
        "busy_period_bias": (float, 0.0, 1.0, "skip quiet periods; act mainly when the market is busy (advisory #5)"),
    },
    "pump_and_dump": {
        "n_coordinated_accounts": (int, 2, 15, "colluding accounts"),
        "accumulation_size": (int, 50, 1000, "target position per account"),
        "burst_duration_ticks": (int, 50, 1000, "accumulate + ignite duration (ignite = last 30%)"),
        "dump_delay_ticks": (int, 10, 500, "passive distribution period before the dump"),
        "stagger": (float, 0.0, 1.0, "per-account random start delay, x burst duration"),
        "ignition_aggression": (float, 0.0002, 0.005, "how far through the ask ignition buys go"),
        "decoy_rate": (float, 0.0, 0.4, "chance per tick of a small opposite-side trade"),
        "busy_period_bias": (float, 0.0, 1.0, "skip quiet periods; act mainly when the market is busy (advisory #5)"),
    },
    "layering": {
        "n_layers": (int, 2, 8, "orders stacked on consecutive levels per cycle (Tower: 3-8)"),
        "layer_spacing": (float, 0.00005, 0.003, "gap between layers as a fraction of price (min one tick)"),
        "layer_size_multiplier": (float, 0.5, 6.0, "layer size vs. average visible level depth"),
        "size_taper": (float, 0.0, 0.9, "0 = equal layers; higher = smaller layers deeper in the book"),
        "cancellation_delay_ticks": (int, 1, 30, "mean ticks the layers rest before being cancelled"),
        "cancel_jitter": (float, 0.0, 0.8, "cancel delay drawn from delay x U(1-j, 1+j)"),
        "frequency": (float, 0.05, 1.0, "chance per tick to start a new layering cycle"),
        "busy_period_bias": (float, 0.0, 1.0, "skip quiet periods; act mainly when the market is busy (advisory #5)"),
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
    "spoofing": {"order_size_multiplier": "down", "price_aggressiveness": "up"},
    "layering": {"layer_spacing": "up", "n_layers": "down", "layer_size_multiplier": "down", "size_taper": "up"},
    "wash_trading": {"trade_frequency": "down", "camouflage_rate": "up"},
    "pump_and_dump": {"ignition_aggression": "down", "accumulation_size": "down",
                      "n_coordinated_accounts": "down", "burst_duration_ticks": "up"},
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
    },
    ["persona", "parameters", "reasoning"],
)
STRATEGIST_TOOLS = [GET_ROUND_METRICS_SCHEMA, GET_SHAP_DRIFT_SCHEMA, GET_ENFORCEMENT_NOTES_SCHEMA, PROPOSE_MUTATION_SCHEMA]

ANALYST_TOOLS = [
    _fn("list_rounds", "List the completed rounds of the current run with their flag counts.", {}),
    _fn(
        "query_flags",
        "Detector flags (post-retrain) for a round, filterable by trader_id and persona. "
        "Returns windows, scores, which layer fired and the top SHAP features.",
        {
            "round_num": {"type": "integer", "description": "Round number; omit for all rounds"},
            "trader_id": {"type": "string"},
            "persona": {"type": "string", "description": "spoofing | wash_trading | pump_and_dump | anomaly"},
            "limit": {"type": "integer", "description": "Max rows to return (default 20)"},
        },
    ),
    _fn(
        "query_round_metrics",
        "Precision/recall/F1/FPR, event detection and latency per persona for a round (pre and post retrain).",
        {"round_num": {"type": "integer"}},
        ["round_num"],
    ),
    _fn(
        "get_trader_history",
        "Summary of one trader's orders and trades in a round's test market, optionally within a tick range.",
        {
            "trader_id": {"type": "string"},
            "round_num": {"type": "integer"},
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

    def propose_mutation(self, persona: str, parameters: dict, reasoning: str = "") -> dict:
        """Validate against PARAM_BOUNDS, then write under the top-level persona key."""
        import yaml
        clean, error = validate_mutation(persona, parameters)
        if error:
            return {"error": f"Validation failed: {error}"}
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
        }

    # ---------------------------------------------------------------- analyst
    def list_rounds(self) -> dict:
        rounds = []
        for r in self._available_rounds():
            m = self._metrics_json(r) or {}
            rounds.append({"round_num": r, "n_flags": m.get("n_flags")})
        if not rounds:
            return {"error": "No completed rounds in this run."}
        return {"rounds": rounds}

    def query_flags(self, round_num: int | None = None, trader_id: str | None = None,
                    persona: str | None = None, limit: int = 10) -> dict:
        rounds = [round_num] if round_num is not None else self._available_rounds()
        frames = []
        for r in rounds:
            d = self._round_dir(r)
            if d is None or not (d / "flags.parquet").exists():
                if round_num is not None:
                    return self._not_found(f"Flags for round {round_num}")
                continue
            df = read_flags(d / "flags.parquet")
            df = df[df["final_flag"]]
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

    def get_trader_history(self, trader_id: str, round_num: int,
                           start_tick: int | None = None, end_tick: int | None = None) -> dict:
        d = self._round_dir(round_num)
        tests = [x for x in ([d / "test"] + sorted(d.glob("test_*")) if d else []) if (x / "order_log.parquet").exists()]
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

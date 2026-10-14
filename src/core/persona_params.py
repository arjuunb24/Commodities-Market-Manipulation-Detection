"""
src/core/persona_params.py
==========================
Single mapping between persona_config.yaml parameter names and SimConfig fields.

persona_config.yaml layout: `round_0` holds the baseline per persona; top-level
persona keys hold the mutation for the next round and are merged over it.
Used by scripts/run_simulation.py (to configure a run) and by the adversarial
orchestrator (to record exactly which parameters each round used).
"""

from __future__ import annotations

from typing import Any

# persona → {yaml parameter: (SimConfig attribute, cast)}
PERSONA_PARAM_MAP: dict[str, dict[str, tuple[str, type]]] = {
    "spoofing": {
        "cancellation_delay_ticks": ("spoof_cancel_delay", int),
        "order_size_multiplier": ("spoof_size_multiplier", float),
        "frequency": ("spoof_frequency", float),
        "price_aggressiveness": ("spoof_aggressiveness", float),
        "size_jitter": ("spoof_size_jitter", float),
        "cancel_jitter": ("spoof_cancel_jitter", float),
        "n_split_orders": ("spoof_n_split", int),
        "n_accounts": ("spoof_n_accounts", int),
        "reprice_to_touch": ("spoof_reprice", int),
        "flip_after_cancel": ("spoof_flip", float),
        "busy_period_bias": ("spoof_busy_bias", float),
        "cover_rate": ("spoof_cover_rate", float),
        "cover_size": ("spoof_cover_size", float),
    },
    "wash_trading": {
        "trade_frequency": ("wash_frequency", float),
        "price_deviation_from_mid": ("wash_price_deviation", float),
        "n_groups": ("wash_n_pairs", int),
        "n_colluding_pairs": ("wash_n_pairs", int),   # legacy name
        "ring_size": ("wash_ring_size", int),
        "size_jitter": ("wash_size_jitter", float),
        "camouflage_rate": ("wash_camouflage_rate", float),
        "busy_period_bias": ("wash_busy_bias", float),
    },
    "pump_and_dump": {
        "burst_duration_ticks": ("pnd_burst_duration", int),
        "n_coordinated_accounts": ("pnd_n_accounts", int),
        "dump_delay_ticks": ("pnd_dump_delay", int),
        "accumulation_size": ("pnd_accumulation_size", int),
        "stagger": ("pnd_stagger", float),
        "ignition_aggression": ("pnd_ignition_aggression", float),
        "decoy_rate": ("pnd_decoy_rate", float),
        "busy_period_bias": ("pnd_busy_bias", float),
    },
    "layering": {
        "n_layers": ("layer_n_layers", int),
        "layer_spacing": ("layer_spacing", float),
        "layer_size_multiplier": ("layer_size_multiplier", float),
        "size_taper": ("layer_size_taper", float),
        "cancellation_delay_ticks": ("layer_cancel_delay", int),
        "cancel_jitter": ("layer_cancel_jitter", float),
        "frequency": ("layer_frequency", float),
        "busy_period_bias": ("layer_busy_bias", float),
        "cover_rate": ("layer_cover_rate", float),
        "cover_size": ("layer_cover_size", float),
    },
}
ENABLE_FLAGS = {
    "spoofing": "enable_spoofing",
    "wash_trading": "enable_wash_trading",
    "pump_and_dump": "enable_pump_and_dump",
    "layering": "enable_layering",
}


def resolve_persona_config(p_cfg: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """round_0 baseline merged with any top-level (mutated) persona parameters."""
    active: dict[str, dict[str, Any]] = {}
    for k, v in (p_cfg.get("round_0") or {}).items():
        if isinstance(v, dict):
            active[k] = dict(v)
    for k, v in p_cfg.items():
        if not k.startswith("round_") and isinstance(v, dict):
            active.setdefault(k, {}).update(v)
    return active


def apply_persona_config(config: Any, active: dict[str, dict[str, Any]]) -> Any:
    """Write resolved persona parameters (and enabled flags) onto a SimConfig."""
    for persona, params in active.items():
        # `enabled` is honoured in both directions (layering is off by default in SimConfig).
        if "enabled" in params and persona in ENABLE_FLAGS:
            setattr(config, ENABLE_FLAGS[persona], bool(params["enabled"]))
        for name, (attr, cast) in PERSONA_PARAM_MAP.get(persona, {}).items():
            if name in params and params[name] is not None:
                setattr(config, attr, cast(params[name]))
    return config


def effective_params(active: dict[str, dict[str, Any]], persona: str) -> dict[str, Any]:
    """The mutable parameters actually in force for a persona (for track records)."""
    known = PERSONA_PARAM_MAP.get(persona, {})
    return {k: v for k, v in (active.get(persona) or {}).items() if k in known and k != "n_colluding_pairs"}

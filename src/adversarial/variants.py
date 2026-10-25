"""
src/adversarial/variants.py
===========================
Randomised persona variants: training diversity and a fixed transfer benchmark.

Each adversarial round used to train on exactly one parameter set per persona (the
one in force that round), so the classifiers learned those settings rather than the
scheme: in run seed1001 a spoofer moved to 3 accounts with cover orders and the
frozen spoofing classifier gave it a median probability of 0.001.

  training variants  every round also simulates a market whose personas start from
                     the round's parameters with several knobs redrawn at random
                     inside PARAM_BOUNDS (domain randomisation). The classifiers then
                     see many neighbours of the current tactic and of untried ones.
  benchmark          a few markets drawn the same way from the round-0 baseline,
                     simulated once per run and never trained on. Every round's
                     retrained detector is scored on them, so "does it generalise to
                     variants it has not seen?" is a curve that can rise over rounds,
                     unlike pre-retrain, which the strategist picks to be the worst case.

Knobs are kept short of known effect cliffs (a spoof further back than the near-touch
band adds no fake depth; layers spaced wider than 0.0004 leave the band; crosses lagged
by more than 2 ticks hand most of the wash volume to strangers), so a variant is still
a working scheme rather than an ineffective one labelled as manipulation.
"""

from __future__ import annotations

from typing import Any

import numpy as np

# Upper limits inside PARAM_BOUNDS where the scheme stops working (see module doc).
VARIANT_LIMITS: dict[str, dict[str, float]] = {
    "spoofing": {"price_aggressiveness": 0.0018},
    "layering": {"layer_spacing": 0.0004},
    "wash_trading": {"cross_lag_ticks": 2, "camouflage_rate": 0.35},
}
KNOBS_PER_VARIANT = (3, 6)


def random_variant(base: dict[str, dict[str, Any]], rng: np.random.Generator,
                   personas: list[str] | None = None) -> dict[str, dict[str, Any]]:
    """
    Persona parameters with KNOBS_PER_VARIANT random knobs per persona redrawn uniformly
    inside PARAM_BOUNDS (and VARIANT_LIMITS); every other knob keeps its `base` value.
    """
    from src.agentic.tools import PARAM_BOUNDS
    out: dict[str, dict[str, Any]] = {}
    for persona, bounds in PARAM_BOUNDS.items():
        if personas is not None and persona not in personas:
            continue
        params = dict(base.get(persona) or {})
        knobs = list(bounds)
        k = int(rng.integers(KNOBS_PER_VARIANT[0], min(len(knobs), KNOBS_PER_VARIANT[1]) + 1))
        for knob in rng.choice(knobs, size=k, replace=False):
            typ, lo, hi, _ = bounds[str(knob)]
            hi = min(hi, VARIANT_LIMITS.get(persona, {}).get(str(knob), hi))
            params[str(knob)] = int(rng.integers(int(lo), int(hi) + 1)) if typ is int \
                else float(f"{rng.uniform(lo, hi):.4g}")
        out[persona] = params
    return out


def variant_config(base: dict[str, dict[str, Any]], rng: np.random.Generator,
                   personas: list[str] | None = None) -> dict:
    """A persona_config.yaml document (round_0 block only) for one random variant."""
    variant = random_variant(base, rng, personas)
    doc = {p: {**(base.get(p) or {}), **v} for p, v in variant.items()}
    for p, v in base.items():   # keep personas without bounds and their `enabled` flags
        doc.setdefault(p, dict(v))
    return {"round_0": doc}

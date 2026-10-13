"""
src/agentic/mutation_library.py
===============================
Hand-authored mutation variants per persona (spec Section 5.2, Section 7.3).

The deterministic fallback when the LLM strategist is unavailable, over budget,
or fails validation twice (FR-4.2 / FR-3.3 / SRS UC-4) — and the baseline the
LLM strategist is compared against. Each progression follows the evasion steps
documented in docs/enforcement_case_notes.md and keeps the scheme effective
(no "evasion by not manipulating").
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

FALLBACK_MUTATIONS: dict[str, list[dict[str, Any]]] = {
    "spoofing": [
        {
            "parameters": {"order_size_multiplier": 6.0, "size_jitter": 0.4, "cancel_jitter": 0.5,
                           "cancellation_delay_ticks": 8},
            "reasoning": (
                "Round 1 (Coscia evasion; advisory #1-2): randomise spoof size (+-40%) and cancel timing "
                "(+-50%, mean 8 ticks) to blur max_order_size_rel and fast_cancel_ratio, keeping the spoof "
                "near the touch so it still moves the book."
            ),
        },
        {
            "parameters": {"order_size_multiplier": 6.0, "size_jitter": 0.5, "cancel_jitter": 0.6,
                           "cancellation_delay_ticks": 10, "n_split_orders": 3},
            "reasoning": (
                "Round 2 (Coscia 'split into batches', Tower layering): split each spoof into 3 orders on "
                "consecutive ticks so no single order is extreme (max_order_size_rel, large_cancel_ratio)."
            ),
        },
        {
            "parameters": {"order_size_multiplier": 6.0, "size_jitter": 0.5, "cancel_jitter": 0.6,
                           "cancellation_delay_ticks": 10, "n_split_orders": 3, "n_accounts": 2},
            "reasoning": (
                "Round 3+ (JPMorgan cross-trader spoofing): the genuine order and the spoof come from different "
                "accounts, breaking the per-account spoof_signature and cancel/execution opposition."
            ),
        },
    ],
    "wash_trading": [
        {
            "parameters": {"size_jitter": 0.6, "ring_size": 3},
            "reasoning": (
                "Round 1 (Coinbase cyclical washes; advisory #1): rotate crosses through a 3-account ring and vary "
                "lot sizes, so no pair dominates (top_cpty_share) and sizes look organic (trade_size_cv)."
            ),
        },
        {
            "parameters": {"size_jitter": 0.6, "ring_size": 3, "n_groups": 2},
            "reasoning": (
                "Round 2 (Kraft round 2; advisory #3): split the same cross rate over two independent groups — "
                "per-account volume share halves while total wash volume is unchanged."
            ),
        },
        {
            "parameters": {"size_jitter": 0.7, "ring_size": 4, "n_groups": 3, "camouflage_rate": 0.2},
            "reasoning": (
                "Round 3+ (Kraft: 'interspersed with small genuine-looking directional trades'; advisory #4): "
                "larger rings, three groups and 20% genuine market trades to break net_flatness/reciprocal_ratio."
            ),
        },
    ],
    "pump_and_dump": [
        {
            "parameters": {"n_coordinated_accounts": 5, "stagger": 0.4, "burst_duration_ticks": 300},
            "reasoning": (
                "Round 1 (Nefedova evasion 1): five accounts with staggered schedules over a longer burst so "
                "their buying does not light up coordinated_peers in the same window."
            ),
        },
        {
            "parameters": {"n_coordinated_accounts": 6, "stagger": 0.6, "burst_duration_ticks": 400,
                           "dump_delay_ticks": 200, "decoy_rate": 0.15},
            "reasoning": (
                "Round 2 (Nefedova evasion 2; advisory #4): phased, longer exit and occasional decoy trades to "
                "soften exec_side_imbalance and reversal_score."
            ),
        },
        {
            "parameters": {"n_coordinated_accounts": 8, "stagger": 0.7, "burst_duration_ticks": 600,
                           "dump_delay_ticks": 300, "decoy_rate": 0.2, "accumulation_size": 120},
            "reasoning": (
                "Round 3+ (Nefedova round 2: 8 accounts, 600-tick accumulation, staggered exit): spread smaller "
                "per-account positions across 8 accounts."
            ),
        },
    ],
}


def get_fallback_mutation(persona: str, round_num: int) -> dict[str, Any]:
    """
    Returns the hand-authored mutation for a given persona and TARGET round
    (round 1 → first variant, 2 → second, ...; later rounds reuse the last,
    most evasive variant instead of wrapping back to the first).
    """
    variants = FALLBACK_MUTATIONS.get(persona, [])
    if not variants:
        logger.warning(f"[MutationLibrary] No hand-authored mutations found for persona '{persona}'.")
        return {"parameters": {}, "reasoning": f"No fallback mutation defined for persona {persona}."}
    idx = min(max(0, round_num - 1), len(variants) - 1)
    return variants[idx]

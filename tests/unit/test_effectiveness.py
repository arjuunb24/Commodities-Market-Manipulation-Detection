"""
tests/unit/test_effectiveness.py
================================
Manipulation effectiveness accounting, the strategist objective, new evasion
knobs and the persona_config → SimConfig mapping.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.adversarial.effectiveness import (
    EFFECTIVENESS_FLOOR,
    effectiveness_ratio,
    episode_effectiveness,
    mutation_score,
    summarize,
)
from src.core.agents.manipulation.spoofing import SpoofingAgent
from src.core.order_book import BookState
from src.core.persona_params import apply_persona_config, effective_params, resolve_persona_config
from src.core.simulation import SimConfig, run_simulation


def _eff(**kw) -> dict:
    rows = []
    for seed in (1, 2):
        out = run_simulation(SimConfig(commodity="crude_oil_wti", n_ticks=4000, write_to_disk=False, **kw), seed)
        rows.append(episode_effectiveness(out.manipulation_events, out.trade_log, out.order_log))
    return summarize(pd.concat(rows, ignore_index=True))


@pytest.fixture(scope="module")
def baseline_eff():
    return _eff()


def test_effectiveness_detects_neutered_manipulation(baseline_eff):
    neutered = _eff(spoof_aggressiveness=0.008, wash_frequency=0.05, pnd_n_accounts=2,
                    pnd_accumulation_size=50, pnd_burst_duration=900)
    for persona in ("spoofing", "wash_trading", "pump_and_dump"):
        ratio = effectiveness_ratio(neutered.get(persona), baseline_eff.get(persona), persona)
        assert ratio is not None and ratio < EFFECTIVENESS_FLOOR, (persona, ratio)


def test_baseline_manipulations_have_an_effect(baseline_eff):
    assert baseline_eff["spoofing"]["favorable_move_bp"] > 0
    assert baseline_eff["spoofing"]["genuine_fill_rate"] > 0.5
    assert baseline_eff["wash_trading"]["wash_volume_share"] > 0.1
    assert baseline_eff["pump_and_dump"]["ignition_pressure"] > 0.05


def test_mutation_score_rewards_only_effective_evasion():
    assert mutation_score(0.2, 1.0) == pytest.approx(0.8)
    assert mutation_score(0.2, 0.7) == pytest.approx(0.8 * 0.7)
    assert mutation_score(0.0, 0.3) == 0.0          # "evasion" by not manipulating
    assert mutation_score(None, 1.0) is None         # no frozen detector in round 0


def test_wash_groups_logged_as_one_episode():
    out = run_simulation(SimConfig(commodity="gold", n_ticks=1500, write_to_disk=False,
                                   enable_spoofing=False, enable_pump_and_dump=False,
                                   wash_start_tick=100, wash_end_tick=900, wash_n_pairs=3, wash_ring_size=3), 3)
    ev = out.manipulation_events
    assert len(ev) == 1 and len(ev.iloc[0]["trader_ids"]) == 9


def test_cross_account_spoofing_uses_different_accounts():
    agent = SpoofingAgent("sp", np.random.default_rng(0), frequency=1.0, n_accounts=2, n_split_orders=3,
                          start_tick=0, end_tick=100)
    state = BookState(best_bid=99.0, best_ask=101.0, mid_price=100.0, bid_depth=[(99.0, 40)],
                      ask_depth=[(101.0, 40)], last_trade_price=100.0, tick=5)
    orders = agent.decide_batch(state, 5)
    genuine = [o for o in orders if o.order_id.endswith("g")]
    spoofs = [o for o in orders if o.order_id.endswith("s")]
    assert len(genuine) == 1 and len(spoofs) == 3
    assert {o.trader_id for o in spoofs} != {genuine[0].trader_id}
    assert len({o.price for o in spoofs}) == 3      # consecutive levels


def test_persona_param_mapping_roundtrip():
    cfg = {"round_0": {"spoofing": {"enabled": True, "frequency": 0.4, "n_accounts": 1},
                       "wash_trading": {"enabled": False}},
           "spoofing": {"n_accounts": 2, "size_jitter": 0.3}}
    active = resolve_persona_config(cfg)
    sim = apply_persona_config(SimConfig(write_to_disk=False), active)
    assert sim.spoof_n_accounts == 2 and sim.spoof_size_jitter == 0.3 and sim.spoof_frequency == 0.4
    assert sim.enable_wash_trading is False
    assert effective_params(active, "spoofing") == {"frequency": 0.4, "n_accounts": 2, "size_jitter": 0.3}

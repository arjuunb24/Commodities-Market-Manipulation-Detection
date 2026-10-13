"""
tests/unit/test_personas.py
===========================
Behavioural tests for the manipulation personas: each one must produce the
footprint of the real tactic it models, inside the market mechanism.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.core.agents.manipulation.pump_and_dump import PumpDumpCoalition, PumpPhase
from src.core.agents.manipulation.spoofing import SpoofingAgent
from src.core.agents.manipulation.wash_trading import WashTradingPair
from src.core.order_book import BookState, Side
from src.core.simulation import SimConfig, run_simulation


def _state(bid=99.0, ask=101.0, tick=10) -> BookState:
    return BookState(best_bid=bid, best_ask=ask, mid_price=(bid + ask) / 2,
                     bid_depth=[(bid, 40)], ask_depth=[(ask, 40)],
                     last_trade_price=100.0, tick=tick)


def test_spoof_cycle_places_genuine_and_large_opposite_orders():
    agent = SpoofingAgent("sp", np.random.default_rng(0), frequency=1.0, spoof_side="buy",
                          start_tick=0, end_tick=100, cancellation_delay_ticks=5)
    orders = agent.decide_batch(_state(), 10)
    genuine, spoof = orders
    assert genuine.side == Side.SELL and genuine.price == 101.0     # rests at the best ask
    assert spoof.side == Side.BUY and spoof.price < 99.0 + 1e-9     # behind the best bid, never crossing
    assert spoof.quantity > 5 * genuine.quantity
    assert agent.decide_batch(_state(), 11) == []                     # one cycle at a time
    cancels = agent.expire_orders(15, lambda oid: True)
    assert spoof.order_id in {c.order_id for c in cancels}


def test_wash_pair_crosses_inside_spread_with_equal_size():
    pair = WashTradingPair.create(np.random.default_rng(1), 0, trade_frequency=1.0,
                                  start_tick=0, end_tick=100)
    leg1, leg2 = pair.decide_batch(_state(), 10)
    assert {leg1.trader_id, leg2.trader_id} == {"wash_0a", "wash_0b"}
    assert leg1.side != leg2.side and leg1.price == leg2.price and leg1.quantity == leg2.quantity
    assert 99.0 < leg1.price < 101.0


def test_wash_pair_waits_when_spread_is_one_tick():
    pair = WashTradingPair.create(np.random.default_rng(1), 0, trade_frequency=1.0,
                                  start_tick=0, end_tick=100)
    assert pair.decide_batch(_state(bid=99.99, ask=100.0), 10) == []


def test_pump_and_dump_phases():
    c = PumpDumpCoalition.create(np.random.default_rng(2), n_coordinated_accounts=3,
                                 burst_duration_ticks=100, dump_delay_ticks=50, start_tick=0, end_tick=10_000)
    expected = {10: PumpPhase.ACCUMULATE, 80: PumpPhase.IGNITE, 120: PumpPhase.DISTRIBUTE,
                160: PumpPhase.DUMP, 400: PumpPhase.IDLE}
    for tick, phase in expected.items():
        c.update_phase(tick)
        assert c.phase == phase, tick
    assert c.ground_truth_end() == 0 + 100 + 50 + 160


@pytest.fixture(scope="module")
def sim():
    return run_simulation(SimConfig(commodity="crude_oil_wti", n_ticks=3000, write_to_disk=False), seed=7)


def test_wash_trades_self_cross_in_simulation(sim):
    tr, ev = sim.trade_log, sim.manipulation_events
    for _, e in ev[ev.persona == "wash_trading"].iterrows():
        ids = set(e.trader_ids)
        w = tr[(tr.tick >= e.start_tick) & (tr.tick <= e.end_tick)
               & (tr.buyer_trader_id.isin(ids) | tr.seller_trader_id.isin(ids))]
        if len(w):
            self_cross = (w.buyer_trader_id.isin(ids) & w.seller_trader_id.isin(ids)).mean()
            assert self_cross > 0.9


def test_manipulator_accounts_have_background_activity(sim):
    od, ev = sim.order_log, sim.manipulation_events
    for _, e in ev.iterrows():
        o = od[od.trader_id.isin(set(e.trader_ids))]
        outside = o[(o.tick < e.start_tick) | (o.tick > e.end_tick)]
        assert len(outside) > 0, f"{e.persona} accounts only exist while manipulating"


def test_episode_trader_ids_unique(sim):
    ids = [t for ts in sim.manipulation_events.trader_ids for t in ts]
    assert len(ids) == len(set(ids))


def test_prices_on_tick_grid(sim):
    ticks = (sim.trade_log.price / 0.01).round(6)
    assert np.allclose(ticks, ticks.round())

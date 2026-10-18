"""
tests/unit/test_replay.py
=========================
Scripted replays of documented real cases: the agent follows its script, the
case library is complete, and replays reach the simulator's ground truth.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.core.agents.manipulation.replay import CaseReplayAgent
from src.core.order_book import BookState, Side
from src.replay.evaluate import load_cases


def _state(bid=100.0, ask=100.1, tick=10, band=1000) -> BookState:
    return BookState(best_bid=bid, best_ask=ask, mid_price=(bid + ask) / 2,
                     bid_depth=[(bid, 100), (bid - 0.1, 100)], ask_depth=[(ask, 100), (ask + 0.1, 100)],
                     last_trade_price=bid, tick=tick, bid_band_volume=band, ask_band_volume=band)


def _agent(script, rng=0, start=0, end=10**6):
    a = CaseReplayAgent("case", script, np.random.default_rng(rng), start_tick=start, end_tick=end)
    for acct in a.accounts:
        acct.tick_size = 0.1
    return a


def _new(orders):
    return [o for o in orders if o.order_type != "cancel"]


def test_case_library_is_complete():
    cases = load_cases()
    assert len(cases) >= 9 and {"sarao_layering_algorithm", "shinhan_wash_trades", "optiver_banging_the_close"} <= set(cases)
    for cid, c in cases.items():
        assert c["label"] in ("spoofing", "layering", "wash_trading", "pump_and_dump"), cid
        assert c["sources"] and c["verified"] and c["approximated"] and c["markets"], cid


def test_sarao_style_layers_target_share_and_distance():
    script = {"label": "layering", "fake": {"side": "sell", "n_orders": 5, "levels_from_touch": 3, "target_share": 0.25,
                                             "lifetime_ticks": [50, 50], "reprice": True}, "genuine": {"place": "none"}}
    a = _agent(script)
    orders = _new(a.manipulate(_state(), 1))
    assert len(orders) == 5 and all(o.side == Side.SELL for o in orders)
    prices = sorted(o.price for o in orders)
    assert prices[0] == pytest.approx(100.1 + 3 * 0.1)                # 3 levels above the best ask
    assert np.allclose(np.diff(prices), 0.1)                            # one level apart
    total = sum(o.quantity for o in orders)
    assert total / (total + 1000) == pytest.approx(0.25, abs=0.01)      # 25% of visible sell depth
    # Touch moves up one tick: every layer is cancelled and re-placed at the same distance
    moved = a.manipulate(_state(bid=100.1, ask=100.2, tick=2), 2)
    assert sum(o.order_type == "cancel" for o in moved) == 5
    assert min(o.price for o in _new(moved)) == pytest.approx(100.2 + 3 * 0.1)


def test_coscia_style_alternates_and_pulls_on_genuine_fill():
    script = {"label": "spoofing", "fake": {"side": "both", "direction_persistence": 0.0, "n_orders": 3,
                                             "levels_from_touch": 0, "size_x_level_depth": 3.0, "lifetime_ticks": [5, 5],
                                             "cancel_on_genuine_fill": True}, "genuine": {"place": "per_cycle", "size_x_level_depth": 0.3},
              "gap_ticks": [0, 0]}
    a = _agent(script)
    first = _new(a.manipulate(_state(), 1))
    genuine, fakes = first[0], first[1:]
    assert len(fakes) == 3 and all(f.side != genuine.side for f in fakes)
    assert all(f.quantity > 5 * genuine.quantity for f in fakes)          # small genuine, large opposite
    a._is_resting = lambda oid: oid != genuine.order_id                    # the genuine order just filled
    pulled = a.manipulate(_state(tick=2), 2)
    assert sum(o.order_type == "cancel" for o in pulled) == 3              # all fakes pulled at once
    second = _new(a.manipulate(_state(tick=3), 3))
    assert second[1].side != fakes[0].side                                 # next cycle reverses direction


def test_program_trade_phases_follow_the_schedule():
    script = {"label": "pump_and_dump", "mode": "program", "phases": [
        {"kind": "trade", "ticks": 10, "side": "buy", "how": "marketable", "size_x_level_depth": 0.5, "rate": 1.0},
        {"kind": "trade", "ticks": 10, "side": "sell", "how": "passive", "size_x_level_depth": 0.5, "rate": 1.0}]}
    a = _agent(script)
    first = [o for t in range(0, 10) for o in a.manipulate(_state(tick=t), t)]
    second = [o for t in range(10, 20) for o in a.manipulate(_state(tick=t), t)]
    after = [o for t in range(20, 25) for o in a.manipulate(_state(tick=t), t)]
    assert first and all(o.side == Side.BUY and o.price == pytest.approx(100.1) for o in first)     # lifts the ask
    assert second and all(o.side == Side.SELL and o.price == pytest.approx(100.1) for o in second)  # rests at its ask
    assert after == []                                                                                # programme over


def test_program_cross_trades_between_own_accounts():
    script = {"label": "wash_trading", "mode": "program", "n_accounts": 2, "phases": [
        {"kind": "cross", "ticks": 50, "size_x_level_depth": 0.5, "rate": 1.0, "cross_delay_ticks": [0, 0]}]}
    a = _agent(script)
    wide = _state(bid=100.0, ask=100.3)
    orders = a.manipulate(wide, 1)
    assert len(orders) == 2
    maker, taker = orders
    assert {maker.trader_id, taker.trader_id} == {acc.trader_id for acc in a.accounts}
    assert maker.side != taker.side and maker.price == taker.price and maker.quantity == taker.quantity
    assert 100.0 < maker.price < 100.3                                       # inside the spread
    assert a.manipulate(_state(bid=100.0, ask=100.1), 2) == []              # one-tick spread: wait


def test_detector_accepts_sample_weights():
    from src.detector.classifiers import SupervisedDetector
    import pandas as pd

    rng = np.random.default_rng(0)
    X = pd.DataFrame({"a": rng.normal(size=400), "b": rng.normal(size=400)})
    y = pd.DataFrame({"spoofing": (X["a"] > 1.0).astype(int)})
    det = SupervisedDetector({"personas": ["spoofing"], "supervised": {"cv_folds": 2}})
    det.fit(X, y, sample_weight=np.where(np.arange(400) < 200, 3.0, 1.0))
    assert "spoofing" in det.models and det.predict_proba(X)["spoofing"].between(0, 1).all()


def test_replay_reaches_simulation_ground_truth():
    from src.core.simulation import SimConfig, run_simulation

    cases = load_cases()
    cfg = SimConfig(commodity="gold", n_ticks=900, write_to_disk=False)
    cfg.enable_spoofing = cfg.enable_wash_trading = cfg.enable_pump_and_dump = cfg.enable_layering = False
    cfg.replay_scripts = [{"case": "jpmorgan_precious_metals", "script": cases["jpmorgan_precious_metals"],
                           "start_tick": 300, "end_tick": 700}]
    out = run_simulation(cfg, 5)
    ev = out.manipulation_events
    assert len(ev) == 1 and ev.iloc[0]["persona"] == "spoofing"
    assert ev.iloc[0]["parameters"]["case"] == "jpmorgan_precious_metals"
    accts = set(ev.iloc[0]["trader_ids"])
    assert len(accts) == 2                                                  # two accounts, as scripted
    placed = out.order_log[(out.order_log["event_type"] == "placed") & out.order_log["trader_id"].isin(accts)]
    assert placed["tick"].between(300, 700).sum() > 10

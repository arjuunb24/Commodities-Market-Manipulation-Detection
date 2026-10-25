"""Legitimate look-alike agents, the frozen depth reference, the new structural knobs and the r0 phase."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.adversarial.metrics import build_summary_table, write_round_metrics_parquet
from src.core.agents.base import DepthReference
from src.core.agents.lookalikes import FundFamily, LadderHedger, QuoteProtectingLP
from src.core.order_book import BookState, Side
from src.core.simulation import SimConfig, run_simulation

PERSONAS = ["spoofing", "wash_trading", "pump_and_dump", "layering"]


def _state(depth: int = 40, tick: int = 1) -> BookState:
    return BookState(best_bid=99.9, best_ask=100.1, mid_price=100.0,
                     bid_depth=[(99.9, depth)], ask_depth=[(100.1, depth)],
                     last_trade_price=100.0, tick=tick)


def test_depth_reference_freezes_after_warmup():
    ref = DepthReference(warmup=5)
    for t in range(5):
        ref.update(_state(40, t))
    assert ref.update(_state(4000, 6)) == pytest.approx(40.0)   # later deep books no longer move it


def test_quote_protecting_lp_pulls_the_opposite_side_when_hit():
    lp = QuoteProtectingLP("lp_t", np.random.default_rng(1), session_prob=1.0)
    lp.axed_prob, lp.refresh_rate, lp.ladder_prob = 0.0, 1.0, 0.0
    orders = []
    for t in range(3):
        orders = lp.decide_batch(_state(tick=t), t)
        if orders:
            break
    sides = {o.side for o in orders}
    assert sides == {Side.BUY, Side.SELL}
    assert max(o.quantity for o in orders) >= 1.5 * 40 - 1          # large size
    lp.update_inventory(Side.BUY, 10)                                # bid hit
    pulled = lp.decide_batch(_state(tick=5), 5)
    assert pulled and all(o.order_type == "cancel" for o in pulled)
    assert len(pulled) == len(orders)


def test_ladder_hedger_places_a_one_sided_ladder_in_one_tick():
    h = LadderHedger("hedger_t", np.random.default_rng(2), start_prob=1.0)
    h.hedge_prob = 0.0
    h.decide_batch(_state(tick=0), 0)                                # starts the program
    ladder = h.decide_batch(_state(tick=1), 1)
    assert len(ladder) >= 2
    assert len({o.side for o in ladder}) == 1 and len({o.price for o in ladder}) == len(ladder)


def test_lookalikes_trade_without_manipulation_labels():
    cfg = SimConfig(commodity="gold", n_ticks=3000, write_to_disk=False,
                    enable_spoofing=False, enable_wash_trading=False, enable_pump_and_dump=False,
                    enable_layering=False, n_noise_traders=8, n_momentum_traders=2,
                    n_mean_reversion_traders=2, n_occasional_traders=0)
    out = run_simulation(cfg, seed=7)
    assert out.manipulation_events.empty
    placed = out.order_log[out.order_log["event_type"] == "placed"]
    for prefix in ("lp_", "hedger_", "fund_"):
        assert placed["trader_id"].str.startswith(prefix).any(), prefix
    # The frozen depth reference keeps sizes bounded (the live average once ran away to 1e16).
    assert placed["quantity"].max() < 50_000


def test_fund_family_spreads_one_plan_over_its_accounts():
    fam = FundFamily("fund_t", np.random.default_rng(3), n_accounts=4, start_prob=1.0)
    traders = set()
    for t in range(400):
        for o in fam.decide_batch(_state(tick=t), t):
            traders.add(o.trader_id)
    assert len(traders) >= 3 and all(tid.startswith("fund_t_") for tid in traders)


def _wash_layer_market(**kw):
    cfg = SimConfig(commodity="gold", n_ticks=2500, write_to_disk=False, enable_spoofing=False,
                    enable_pump_and_dump=False, enable_layering=True, enable_wash_trading=True,
                    wash_start_tick=300, wash_end_tick=1000, layer_start_tick=1300, layer_end_tick=2000, **kw)
    return run_simulation(cfg, seed=3)


def test_layering_can_spread_the_ladder_over_accounts():
    out = _wash_layer_market(layer_n_accounts=3)
    ev = out.manipulation_events
    accts = list(ev[ev["persona"] == "layering"]["trader_ids"].iloc[0])
    assert len(accts) == 3
    layers = out.order_log[(out.order_log["event_type"] == "placed") & out.order_log["order_id"].str.endswith("l")]
    assert set(layers["trader_id"]) == set(accts[1:])               # genuine account places no layers


def test_wash_cross_lag_lets_strangers_into_the_ring_trades():
    from src.adversarial.effectiveness import episode_effectiveness, summarize
    share = {}
    for lag in (0, 2):
        out = _wash_layer_market(wash_cross_lag=lag)
        share[lag] = summarize(episode_effectiveness(out.manipulation_events, out.trade_log, out.order_log))[
            "wash_trading"]["wash_volume_share"]
    assert 0 < share[2] < share[0]


def test_summary_has_r0_phase_and_caught_coverage(tmp_path):
    m = {p: {"precision": 0.9, "recall": 0.8, "f1": 0.85, "fpr": 0.0, "support": 5, "events_total": 1,
             "events_detected": 1, "caught_coverage": 0.7} for p in PERSONAS}
    df = build_summary_table([{"round": 2, "personas": m, "pre_retrain": m, "round0_detector": m}])
    assert set(df["phase"]) == {"post", "pre", "r0"}
    assert (df["caught_coverage"] == 0.7).all()
    write_round_metrics_parquet(df, "all", tmp_path / "round_metrics.parquet")
    rm = pd.read_parquet(tmp_path / "round_metrics.parquet")
    assert len(rm) == 2 * len(PERSONAS)                                  # r0 rows stay out of the SRS table

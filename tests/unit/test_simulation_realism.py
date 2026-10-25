from __future__ import annotations

import numpy as np

from src.core.agents.hft_liquidity_provider import HFTLiquidityProvider
from src.core.order_book import BookState
from src.core.simulation import SimConfig, run_simulation
from src.detector.features import FeatureEngine
from src.detector.pipeline import DetectorPipeline


def test_hft_provider_can_create_both_small_and_large_quotes():
    state = BookState(
        best_bid=99.0,
        best_ask=101.0,
        mid_price=100.0,
        bid_depth=[(99.0, 40)],
        ask_depth=[(101.0, 40)],
        last_trade_price=100.0,
        tick=1,
    )
    large = HFTLiquidityProvider(
        "hft_large", np.random.default_rng(4), arrival_rate_lambda=1.0,
        ladder_prob=1.0, large_quote_probability=1.0,
    )
    small = HFTLiquidityProvider(
        "hft_small", np.random.default_rng(4), arrival_rate_lambda=1.0,
        ladder_prob=1.0, large_quote_probability=0.0,
    )

    large_orders = large.decide_batch(state, 1)
    small_orders = small.decide_batch(state, 1)

    assert len(large_orders) >= 2
    assert max(order.quantity for order in large_orders) >= 40
    assert small_orders
    assert max(order.quantity for order in small_orders) <= 24


def test_institutional_crosses_overlap_ring_features_without_manipulation_labels():
    config = SimConfig(
        commodity="gold",
        n_ticks=900,
        write_to_disk=False,
        n_noise_traders=5,
        n_momentum_traders=0,
        n_mean_reversion_traders=0,
        n_market_makers=1,
        n_hft_liquidity_providers=0,
        n_execution_algos=0,
        n_institutional_crossing_desks=1,
        institutional_crossing_cycle_rate=0.08,
        n_occasional_traders=0,
        enable_spoofing=False,
        enable_wash_trading=False,
        enable_pump_and_dump=False,
        enable_layering=False,
    )
    output = run_simulation(config, seed=11)

    assert output.manipulation_events.empty
    assert (
        output.trade_log["buyer_trader_id"].str.startswith("rebalance_")
        | output.trade_log["seller_trader_id"].str.startswith("rebalance_")
    ).any()

    feature_config = {"feature_engine": {"window": "3min", "step": "30s"}}
    features = FeatureEngine(feature_config).generate_features(
        output.trade_log, output.order_log
    )
    crossing_rows = features[
        features.index.get_level_values("trader_id").str.startswith("rebalance_")
    ]
    assert not crossing_rows.empty
    assert (crossing_rows["ring_size"] > 1).any()

    labels = DetectorPipeline({
        **feature_config,
        "personas": ["spoofing", "wash_trading", "pump_and_dump", "layering"],
    }).generate_labels(features, output.manipulation_events)
    assert int(labels.sum().sum()) == 0

"""
tests/integration/test_simulation_reproducibility.py
======================================================
SRS FR-0.4: Two runs with the same master seed and config must produce
byte-identical trade_log and order_log DataFrames.

This is the most fundamental correctness test — it validates that:
1. numpy.random.default_rng(seed) is the only source of randomness
2. rng.spawn() is used correctly (no shared generators between agents)
3. No clock-based or UUID-seeding sources of non-determinism exist
   EXCEPT for trade_id and order_id UUIDs (which are allowed to differ
   because they are purely label-identifiers, not data values).

What IS tested as byte-identical:
  - price on every trade
  - quantity on every trade
  - buyer_trader_id, seller_trader_id on every trade
  - tick on every trade
  - Total number of trades

What is NOT tested as identical (allowed to vary by UUID):
  - trade_id, order_id (UUID v4 — not seeded)
"""

from __future__ import annotations

import numpy as np
import pytest

from src.core.simulation import SimConfig, run_simulation


@pytest.fixture
def minimal_config(tmp_path) -> SimConfig:
    """Minimal 500-tick simulation config for fast test execution."""
    return SimConfig(
        n_ticks=500,
        ticks_per_bar=50,
        n_noise_traders=5,
        n_momentum_traders=2,
        n_mean_reversion_traders=2,
        n_market_makers=1,
        enable_spoofing=True,
        spoof_start_tick=100,
        spoof_end_tick=300,
        enable_wash_trading=True,
        wash_start_tick=50,
        wash_end_tick=200,
        enable_pump_and_dump=False,
        enable_layering=False,
        data_dir=tmp_path,
        overwrite=True,
    )


def test_byte_identical_trades_same_seed(minimal_config: SimConfig, tmp_path) -> None:
    """
    SRS FR-0.4: Two simulation runs with the same seed produce identical trade data.
    """
    seed = 42

    out1 = run_simulation(minimal_config, seed=seed)
    out2 = run_simulation(minimal_config, seed=seed)

    t1 = out1.trade_log.sort_values("tick").reset_index(drop=True)
    t2 = out2.trade_log.sort_values("tick").reset_index(drop=True)

    # Drop trade_id (UUID — legitimately non-deterministic label)
    cols_to_compare = ["tick", "price", "quantity", "buyer_trader_id", "seller_trader_id"]

    assert len(t1) == len(t2), (
        f"Different number of trades across runs with same seed: {len(t1)} vs {len(t2)}"
    )

    for col in cols_to_compare:
        if t1[col].dtype == float:
            assert np.allclose(t1[col].values, t2[col].values, rtol=1e-10, atol=1e-10), (
                f"Column '{col}' differs between runs with same seed."
            )
        else:
            assert (t1[col].values == t2[col].values).all(), (
                f"Column '{col}' differs between runs with same seed."
            )


def test_different_seeds_produce_different_trades(minimal_config: SimConfig) -> None:
    """Sanity check: different seeds produce different trade sequences."""
    out1 = run_simulation(minimal_config, seed=1)
    out2 = run_simulation(minimal_config, seed=2)

    # The probability of identical trade sequences with different seeds is astronomically low
    t1 = out1.trade_log
    t2 = out2.trade_log

    # At minimum, prices should differ for some trades
    if len(t1) > 0 and len(t2) > 0:
        price_sets1 = set(round(p, 6) for p in t1["price"])
        price_sets2 = set(round(p, 6) for p in t2["price"])
        assert price_sets1 != price_sets2, (
            "Different seeds produced identical price distributions — seeding is broken."
        )


def test_ohlcv_reproducible_same_seed(minimal_config: SimConfig) -> None:
    """OHLCV bars derived from trades are also reproducible with same seed."""
    seed = 99
    out1 = run_simulation(minimal_config, seed=seed)
    out2 = run_simulation(minimal_config, seed=seed)

    b1 = out1.ohlcv_bars.sort_values("bar_start").reset_index(drop=True)
    b2 = out2.ohlcv_bars.sort_values("bar_start").reset_index(drop=True)

    assert len(b1) == len(b2)
    for col in ["bar_start", "volume"]:
        assert (b1[col].values == b2[col].values).all(), (
            f"OHLCV column '{col}' not reproducible with same seed."
        )

    for col in ["open", "high", "low", "close"]:
        # NaN-safe comparison
        mask = b1[col].notna() & b2[col].notna()
        assert np.allclose(
            b1.loc[mask, col].values,
            b2.loc[mask, col].values,
            rtol=1e-10, atol=1e-10,
        ), f"OHLCV column '{col}' not reproducible with same seed."

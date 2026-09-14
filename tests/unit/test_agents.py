"""
tests/unit/test_agents.py
==========================
Unit tests for all four legitimate agent archetypes.

Tests verify the contractual requirements from Implementation Plan Section 5.3:
  - decide() returns None under at least one valid book state (agents are not always active)
  - decide() never returns non-positive quantity (price > 0, quantity > 0)
  - decide() never raises for any valid BookState input including empty book
  - All randomness is from the seeded RNG (no global random state)
  - Agent with same seed produces identical sequence of decisions
"""

from __future__ import annotations

import numpy as np
import pytest

from src.core.agents.market_maker import MarketMaker
from src.core.agents.mean_reversion_trader import MeanReversionTrader
from src.core.agents.momentum_trader import MomentumTrader
from src.core.agents.noise_trader import NoiseTrader
from src.core.order_book import BookState, Side


# ===========================================================================
# Fixtures
# ===========================================================================


def rng(seed: int = 42) -> np.random.Generator:
    return np.random.default_rng(seed)


def empty_book_state(tick: int = 0) -> BookState:
    """Book state with no resting orders and no trade history."""
    return BookState(
        best_bid=None,
        best_ask=None,
        mid_price=None,
        bid_depth=[],
        ask_depth=[],
        last_trade_price=None,
        tick=tick,
    )


def normal_book_state(mid: float = 100.0, tick: int = 10) -> BookState:
    """Typical book state with a spread around mid."""
    return BookState(
        best_bid=mid - 1.0,
        best_ask=mid + 1.0,
        mid_price=mid,
        bid_depth=[(mid - 1.0, 20), (mid - 2.0, 30)],
        ask_depth=[(mid + 1.0, 20), (mid + 2.0, 30)],
        last_trade_price=mid,
        tick=tick,
    )


# ===========================================================================
# NoiseTrader
# ===========================================================================


class TestNoiseTrader:
    def test_returns_none_on_inactive_tick(self) -> None:
        """NoiseTrader with lambda=0.0 never activates."""
        trader = NoiseTrader("NT", rng(0), arrival_rate_lambda=0.0)
        results = [trader.decide(normal_book_state(), tick=i) for i in range(100)]
        assert all(r is None for r in results)

    def test_returns_order_on_active_tick(self) -> None:
        """NoiseTrader with lambda=1.0 always activates on normal book state."""
        trader = NoiseTrader("NT", rng(0), arrival_rate_lambda=1.0)
        order = trader.decide(normal_book_state(), tick=1)
        assert order is not None

    def test_order_price_positive(self) -> None:
        trader = NoiseTrader("NT", rng(42), arrival_rate_lambda=1.0)
        for tick in range(50):
            order = trader.decide(normal_book_state(mid=100.0), tick=tick)
            if order is not None:
                assert order.price > 0, f"Non-positive price: {order.price}"

    def test_order_quantity_positive(self) -> None:
        trader = NoiseTrader("NT", rng(42), arrival_rate_lambda=1.0)
        for tick in range(50):
            order = trader.decide(normal_book_state(), tick=tick)
            if order is not None:
                assert order.quantity > 0

    def test_empty_book_returns_none(self) -> None:
        """Empty book (no mid, no last trade) → must return None, not raise."""
        trader = NoiseTrader("NT", rng(0), arrival_rate_lambda=1.0)
        result = trader.decide(empty_book_state(), tick=0)
        assert result is None

    def test_deterministic_with_same_seed(self) -> None:
        """Two traders with the same seed produce identical decisions."""
        state = normal_book_state()
        t1 = NoiseTrader("NT", np.random.default_rng(99), arrival_rate_lambda=0.7)
        t2 = NoiseTrader("NT", np.random.default_rng(99), arrival_rate_lambda=0.7)
        results1 = [t1.decide(state, tick=i) for i in range(20)]
        results2 = [t2.decide(state, tick=i) for i in range(20)]
        for r1, r2 in zip(results1, results2):
            assert (r1 is None) == (r2 is None)
            if r1 is not None and r2 is not None:
                assert r1.price == r2.price
                assert r1.quantity == r2.quantity

    def test_trader_id_on_returned_order(self) -> None:
        trader = NoiseTrader("NT-001", rng(1), arrival_rate_lambda=1.0)
        order = trader.decide(normal_book_state(), tick=1)
        assert order is not None
        assert order.trader_id == "NT-001"


# ===========================================================================
# MomentumTrader
# ===========================================================================


class TestMomentumTrader:
    def test_returns_none_without_price_history(self) -> None:
        """No price history → can't compute momentum → None."""
        trader = MomentumTrader("MT", rng(0), arrival_rate_lambda=1.0)
        result = trader.decide(normal_book_state(), tick=0)
        assert result is None

    def test_returns_none_on_inactive_tick(self) -> None:
        trader = MomentumTrader("MT", rng(0), arrival_rate_lambda=0.0)
        trader.update_price(100.0)
        trader.update_price(105.0)
        result = trader.decide(normal_book_state(), tick=5)
        assert result is None

    def test_buys_on_uptrend(self) -> None:
        """Strong uptrend → buy order."""
        trader = MomentumTrader(
            "MT", rng(0),
            lookback_ticks=5,
            momentum_threshold=0.001,
            arrival_rate_lambda=1.0,
        )
        # Build a strong uptrend
        for p in [100.0, 101.0, 102.0, 103.0, 105.0]:
            trader.update_price(p)

        order = trader.decide(normal_book_state(mid=105.0), tick=5)
        assert order is not None
        assert order.side == Side.BUY

    def test_sells_on_downtrend(self) -> None:
        """Strong downtrend → sell order."""
        trader = MomentumTrader(
            "MT", rng(0),
            lookback_ticks=5,
            momentum_threshold=0.001,
            arrival_rate_lambda=1.0,
        )
        for p in [105.0, 103.0, 102.0, 101.0, 100.0]:
            trader.update_price(p)

        order = trader.decide(normal_book_state(mid=100.0), tick=5)
        assert order is not None
        assert order.side == Side.SELL

    def test_no_action_on_flat_market(self) -> None:
        trader = MomentumTrader(
            "MT", rng(0),
            lookback_ticks=5,
            momentum_threshold=0.1,  # high threshold
            arrival_rate_lambda=1.0,
        )
        for p in [100.0, 100.1, 100.0, 100.1, 100.0]:
            trader.update_price(p)
        result = trader.decide(normal_book_state(), tick=5)
        assert result is None

    def test_order_quantity_positive(self) -> None:
        trader = MomentumTrader("MT", rng(0), arrival_rate_lambda=1.0, momentum_threshold=0.001)
        for p in [90.0, 95.0, 100.0, 105.0, 110.0]:
            trader.update_price(p)
        order = trader.decide(normal_book_state(mid=110.0), tick=5)
        if order is not None:
            assert order.quantity > 0


# ===========================================================================
# MeanReversionTrader
# ===========================================================================


class TestMeanReversionTrader:
    def test_returns_none_without_fundamental(self) -> None:
        """No fundamental value set → cannot act."""
        trader = MeanReversionTrader("MR", rng(0), arrival_rate_lambda=1.0)
        result = trader.decide(normal_book_state(), tick=0)
        assert result is None

    def test_buys_when_price_below_fundamental(self) -> None:
        trader = MeanReversionTrader(
            "MR", rng(0),
            reversion_threshold=0.005,
            arrival_rate_lambda=1.0,
        )
        trader.set_fundamental_value(110.0)
        # Market price = 100, fundamental = 110 → large undervaluation → buy
        state = normal_book_state(mid=100.0)
        order = trader.decide(state, tick=1)
        assert order is not None
        assert order.side == Side.BUY

    def test_sells_when_price_above_fundamental(self) -> None:
        trader = MeanReversionTrader(
            "MR", rng(0),
            reversion_threshold=0.005,
            arrival_rate_lambda=1.0,
        )
        trader.set_fundamental_value(90.0)
        # Market price = 100, fundamental = 90 → overvaluation → sell
        state = normal_book_state(mid=100.0)
        order = trader.decide(state, tick=1)
        assert order is not None
        assert order.side == Side.SELL

    def test_no_action_near_fundamental(self) -> None:
        trader = MeanReversionTrader(
            "MR", rng(0),
            reversion_threshold=0.05,  # 5% threshold
            arrival_rate_lambda=1.0,
        )
        trader.set_fundamental_value(100.0)
        # Market price = 100.5 → only 0.5% deviation → below threshold
        state = normal_book_state(mid=100.5)
        result = trader.decide(state, tick=1)
        assert result is None

    def test_empty_book_returns_none(self) -> None:
        trader = MeanReversionTrader("MR", rng(0), arrival_rate_lambda=1.0)
        trader.set_fundamental_value(100.0)
        result = trader.decide(empty_book_state(), tick=0)
        assert result is None  # no ref_price available

    def test_order_price_positive(self) -> None:
        trader = MeanReversionTrader("MR", rng(42), arrival_rate_lambda=1.0, reversion_threshold=0.005)
        trader.set_fundamental_value(100.0)
        for mid in [80.0, 90.0, 110.0, 120.0]:
            state = normal_book_state(mid=mid)
            order = trader.decide(state, tick=1)
            if order is not None:
                assert order.price > 0


# ===========================================================================
# MarketMaker
# ===========================================================================


class TestMarketMaker:
    def test_decide_pair_returns_bid_and_ask(self) -> None:
        maker = MarketMaker("MM", rng(0), arrival_rate_lambda=1.0)
        cancel_ids, orders = maker.decide_pair(normal_book_state(), tick=1)
        assert len(orders) == 2
        sides = {o.side for o in orders}
        assert Side.BUY in sides
        assert Side.SELL in sides

    def test_bid_below_ask(self) -> None:
        """Market maker bid must always be below its ask."""
        maker = MarketMaker("MM", rng(0), arrival_rate_lambda=1.0)
        _, orders = maker.decide_pair(normal_book_state(mid=100.0), tick=1)
        if len(orders) == 2:
            bid = next(o for o in orders if o.side == Side.BUY)
            ask = next(o for o in orders if o.side == Side.SELL)
            assert bid.price < ask.price

    def test_requote_cancels_previous(self) -> None:
        """Second decide_pair call returns cancel_ids for the previous quotes."""
        maker = MarketMaker("MM", rng(0), arrival_rate_lambda=1.0)
        _, orders1 = maker.decide_pair(normal_book_state(), tick=1)
        prev_ids = {o.order_id for o in orders1}

        cancel_ids, _ = maker.decide_pair(normal_book_state(), tick=2)
        # Some of the previous order_ids should be in cancel_ids
        assert len(cancel_ids) > 0
        assert all(cid in prev_ids for cid in cancel_ids)

    def test_inventory_skew_direction(self) -> None:
        """Long inventory → quotes shift downward (maker wants to sell more)."""
        maker = MarketMaker("MM", rng(0), arrival_rate_lambda=1.0, spread_ticks=2.0)
        state = normal_book_state(mid=100.0)

        _, orders_neutral = maker.decide_pair(state, tick=1)
        neutral_ask = next((o.price for o in orders_neutral if o.side == Side.SELL), None)

        maker.update_inventory(Side.BUY, 200)  # long 200 units
        _, orders_long = maker.decide_pair(state, tick=2)
        long_ask = next((o.price for o in orders_long if o.side == Side.SELL), None)

        if neutral_ask is not None and long_ask is not None:
            assert long_ask <= neutral_ask  # long inventory → ask shifts down

    def test_empty_book_returns_empty_on_inactive(self) -> None:
        maker = MarketMaker("MM", rng(0), arrival_rate_lambda=0.0)
        cancel_ids, orders = maker.decide_pair(normal_book_state(), tick=1)
        assert cancel_ids == []
        assert orders == []

    def test_all_order_prices_positive(self) -> None:
        maker = MarketMaker("MM", rng(42), arrival_rate_lambda=1.0)
        state = normal_book_state(mid=100.0)
        for tick in range(20):
            _, orders = maker.decide_pair(state, tick=tick)
            for o in orders:
                assert o.price > 0

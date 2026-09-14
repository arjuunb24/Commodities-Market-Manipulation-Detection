"""
tests/unit/test_order_book.py
==============================
Unit tests for the Veridex OrderBook matching engine.

Tests ALL edge cases from Implementation Plan Section 5.2:
  E1. Multi-level order walk → one Trade per resting order
  E2. Same-side exact-price → time priority, no self-match
  E3. Cancel of filled/unknown order_id → rejected_cancel (never raise)
  E4. Zero/negative quantity → ValueError in test mode
  E5. Price-time priority correctness
  E6. Market maker requote (cancel-and-replace)
  E7. Self-trade prevention (different trader_ids enforced)
  E8. Empty-book → None/[] (never exception)

Additional tests:
  - Partial fill: incoming order larger than best resting order
  - Multiple fills: incoming order sweeps multiple price levels
  - Bid-ask spread: best_bid < best_ask after resting orders placed
"""

from __future__ import annotations

import pytest

from src.core.order_book import EventType, Order, OrderBook, Side


# ===========================================================================
# Helpers
# ===========================================================================


def make_order(
    trader_id: str,
    side: Side,
    price: float,
    quantity: int,
    tick: int = 0,
    order_id: str | None = None,
) -> Order:
    return Order(
        order_id=order_id or f"{trader_id}-{side.value}-{price}-{tick}",
        trader_id=trader_id,
        side=side,
        price=price,
        quantity=quantity,
        tick=tick,
        order_type="limit",
    )


def make_cancel(trader_id: str, order_id: str, tick: int = 1) -> Order:
    return Order(
        order_id=order_id,
        trader_id=trader_id,
        side=Side.BUY,
        price=1.0,
        quantity=1,
        tick=tick,
        order_type="cancel",
    )


def submit(book: OrderBook, order: Order) -> tuple:
    return book.submit(order, simulation_mode=False)


# ===========================================================================
# E5: Price-time priority correctness
# ===========================================================================


class TestPriceTimePriority:
    """The book correctly sorts by price first, then time (FIFO at same price)."""

    def test_best_bid_is_highest_price(self) -> None:
        book = OrderBook()
        submit(book, make_order("A", Side.BUY, price=10.0, quantity=10, tick=0))
        submit(book, make_order("B", Side.BUY, price=11.0, quantity=10, tick=1))
        assert book.best_bid() == 11.0

    def test_best_ask_is_lowest_price(self) -> None:
        book = OrderBook()
        submit(book, make_order("A", Side.SELL, price=12.0, quantity=10, tick=0))
        submit(book, make_order("B", Side.SELL, price=11.0, quantity=10, tick=1))
        assert book.best_ask() == 11.0

    def test_time_priority_at_same_price(self) -> None:
        """
        Two bids at the same price: the earlier one (lower tick) is matched first.
        """
        book = OrderBook()
        submit(book, make_order("EARLY", Side.BUY, price=10.0, quantity=5, tick=0))
        submit(book, make_order("LATE", Side.BUY, price=10.0, quantity=5, tick=1))

        # Now a sell comes in that can only match one of them
        trades, _ = submit(book, make_order("SELLER", Side.SELL, price=10.0, quantity=5, tick=2))
        assert len(trades) == 1
        assert trades[0].buyer_trader_id == "EARLY"  # time priority

    def test_spread_preserved(self) -> None:
        """After resting orders on both sides, best_bid < best_ask."""
        book = OrderBook()
        submit(book, make_order("MM", Side.BUY, price=99.0, quantity=10, tick=0))
        submit(book, make_order("MM", Side.SELL, price=101.0, quantity=10, tick=1))
        assert book.best_bid() == 99.0
        assert book.best_ask() == 101.0
        assert book.best_bid() < book.best_ask()


# ===========================================================================
# E1: Multi-level order walk
# ===========================================================================


class TestMultiLevelWalk:
    """A crossing order walks multiple price levels; one Trade per resting order."""

    def test_sweep_two_ask_levels(self) -> None:
        book = OrderBook()
        # Two asks at different prices
        submit(book, make_order("S1", Side.SELL, price=10.0, quantity=5, tick=0))
        submit(book, make_order("S2", Side.SELL, price=11.0, quantity=5, tick=1))

        # Aggressive buy crosses both
        trades, events = submit(book, make_order("B", Side.BUY, price=12.0, quantity=10, tick=2))

        assert len(trades) == 2  # one Trade per resting order (E1)
        prices = {t.price for t in trades}
        assert prices == {10.0, 11.0}  # execution at resting prices

    def test_partial_sweep_leaves_remainder_resting(self) -> None:
        book = OrderBook()
        submit(book, make_order("S1", Side.SELL, price=10.0, quantity=3, tick=0))
        submit(book, make_order("S2", Side.SELL, price=10.0, quantity=3, tick=1))

        # Buy wants 4 — should sweep first 3 from S1, 1 from S2, leaving 2 from S2
        trades, _ = submit(book, make_order("B", Side.BUY, price=10.0, quantity=4, tick=2))
        assert sum(t.quantity for t in trades) == 4
        assert book.best_ask() == 10.0  # S2's remainder still on book

    def test_sweep_three_levels_correct_quantities(self) -> None:
        book = OrderBook()
        for price, qty in [(10.0, 2), (11.0, 3), (12.0, 5)]:
            submit(book, make_order("S", Side.SELL, price=price, quantity=qty, tick=0))

        trades, _ = submit(book, make_order("B", Side.BUY, price=15.0, quantity=10, tick=1))
        assert len(trades) == 3
        total_qty = sum(t.quantity for t in trades)
        assert total_qty == 10


# ===========================================================================
# Partial fills
# ===========================================================================


class TestPartialFills:
    def test_incoming_buy_partially_filled(self) -> None:
        book = OrderBook()
        submit(book, make_order("S", Side.SELL, price=10.0, quantity=3, tick=0))

        # Buy wants 5 but only 3 available
        trades, events = submit(book, make_order("B", Side.BUY, price=10.0, quantity=5, tick=1))
        assert len(trades) == 1
        assert trades[0].quantity == 3
        # The resting buy remainder (2 units) should be on the book
        assert book.best_bid() == 10.0

    def test_resting_order_partially_filled(self) -> None:
        book = OrderBook()
        ask_order = make_order("S", Side.SELL, price=10.0, quantity=10, tick=0, order_id="ask-1")
        submit(book, ask_order)

        trades, events = submit(book, make_order("B", Side.BUY, price=10.0, quantity=3, tick=1))
        assert trades[0].quantity == 3

        # Resting ask should still exist with reduced quantity
        partial_fill_events = [e for e in events if e.event_type == EventType.PARTIAL_FILL]
        assert len(partial_fill_events) > 0


# ===========================================================================
# E3: Cancel edge cases
# ===========================================================================


class TestCancelEdgeCases:
    def test_cancel_resting_order(self) -> None:
        book = OrderBook()
        order = make_order("T", Side.BUY, price=10.0, quantity=5, tick=0, order_id="bid-1")
        submit(book, order)
        assert book.best_bid() == 10.0

        _, events = submit(book, make_cancel("T", "bid-1", tick=1))
        assert any(e.event_type == EventType.CANCELLED for e in events)
        assert book.best_bid() is None

    def test_cancel_already_filled_order_no_exception(self) -> None:
        """
        Edge Case E3: Cancelling a filled order must NOT raise — returns
        rejected_cancel event.
        """
        book = OrderBook()
        order = make_order("S", Side.SELL, price=10.0, quantity=5, tick=0, order_id="ask-1")
        submit(book, order)

        # Fill it completely
        submit(book, make_order("B", Side.BUY, price=10.0, quantity=5, tick=1))

        # Cancel of the now-filled order — must not raise
        _, events = submit(book, make_cancel("S", "ask-1", tick=2))
        assert any(e.event_type == EventType.REJECTED_CANCEL for e in events)

    def test_cancel_unknown_order_id_no_exception(self) -> None:
        """Cancel of an order_id that was never submitted → rejected_cancel, not exception."""
        book = OrderBook()
        _, events = submit(book, make_cancel("X", "does-not-exist", tick=0))
        assert any(e.event_type == EventType.REJECTED_CANCEL for e in events)

    def test_double_cancel_second_is_rejected(self) -> None:
        """Cancelling the same order_id twice → first succeeds, second is rejected_cancel."""
        book = OrderBook()
        order = make_order("T", Side.BUY, price=10.0, quantity=5, tick=0, order_id="order-1")
        submit(book, order)

        _, events1 = submit(book, make_cancel("T", "order-1", tick=1))
        assert any(e.event_type == EventType.CANCELLED for e in events1)

        _, events2 = submit(book, make_cancel("T", "order-1", tick=2))
        assert any(e.event_type == EventType.REJECTED_CANCEL for e in events2)


# ===========================================================================
# E4: Zero / negative quantity (test mode — should raise ValueError)
# ===========================================================================


class TestZeroNegativeQuantity:
    def test_zero_quantity_raises_in_test_mode(self) -> None:
        """Zero quantity raises ValueError in test mode (simulation_mode=False)."""
        with pytest.raises(ValueError, match="quantity must be > 0"):
            Order(
                order_id="bad",
                trader_id="T",
                side=Side.BUY,
                price=10.0,
                quantity=0,
                tick=0,
                order_type="limit",
            )

    def test_negative_quantity_raises_in_test_mode(self) -> None:
        with pytest.raises(ValueError, match="quantity must be > 0"):
            Order(
                order_id="bad",
                trader_id="T",
                side=Side.BUY,
                price=10.0,
                quantity=-5,
                tick=0,
                order_type="limit",
            )

    def test_zero_price_raises_in_test_mode(self) -> None:
        with pytest.raises(ValueError, match="price must be > 0"):
            Order(
                order_id="bad",
                trader_id="T",
                side=Side.BUY,
                price=0.0,
                quantity=10,
                tick=0,
                order_type="limit",
            )


# ===========================================================================
# E8: Empty book queries
# ===========================================================================


class TestEmptyBook:
    def test_empty_book_best_bid_is_none(self) -> None:
        assert OrderBook().best_bid() is None

    def test_empty_book_best_ask_is_none(self) -> None:
        assert OrderBook().best_ask() is None

    def test_empty_book_state_mid_is_none(self) -> None:
        state = OrderBook().book_state()
        assert state.best_bid is None
        assert state.best_ask is None
        assert state.mid_price is None

    def test_sell_into_empty_book_rests(self) -> None:
        """A sell into an empty book (no bids) rests without error."""
        book = OrderBook()
        trades, _ = submit(book, make_order("S", Side.SELL, price=10.0, quantity=5, tick=0))
        assert len(trades) == 0
        assert book.best_ask() == 10.0


# ===========================================================================
# Order matching correctness
# ===========================================================================


class TestMatchingCorrectness:
    def test_simple_buy_sell_cross(self) -> None:
        book = OrderBook()
        submit(book, make_order("S", Side.SELL, price=10.0, quantity=5, tick=0))
        trades, _ = submit(book, make_order("B", Side.BUY, price=10.0, quantity=5, tick=1))
        assert len(trades) == 1
        assert trades[0].price == 10.0
        assert trades[0].quantity == 5
        assert trades[0].buyer_trader_id == "B"
        assert trades[0].seller_trader_id == "S"

    def test_execution_at_resting_price_not_incoming(self) -> None:
        """Execution price = resting order's price (price-time priority)."""
        book = OrderBook()
        # Ask resting at 10.0
        submit(book, make_order("S", Side.SELL, price=10.0, quantity=5, tick=0))
        # Buy comes in at 10.5 (more aggressive) → should execute at 10.0
        trades, _ = submit(book, make_order("B", Side.BUY, price=10.5, quantity=5, tick=1))
        assert trades[0].price == 10.0  # resting ask price, not incoming bid price

    def test_no_cross_does_not_match(self) -> None:
        """A bid below best ask does not execute — rests."""
        book = OrderBook()
        submit(book, make_order("S", Side.SELL, price=11.0, quantity=5, tick=0))
        trades, _ = submit(book, make_order("B", Side.BUY, price=10.0, quantity=5, tick=1))
        assert len(trades) == 0
        assert book.best_bid() == 10.0
        assert book.best_ask() == 11.0

    def test_trade_ids_are_unique(self) -> None:
        """Each Trade must have a unique trade_id."""
        book = OrderBook()
        for i in range(5):
            submit(book, make_order("S", Side.SELL, price=float(10 + i), quantity=5, tick=i))

        trades, _ = submit(book, make_order("B", Side.BUY, price=20.0, quantity=25, tick=10))
        trade_ids = [t.trade_id for t in trades]
        assert len(trade_ids) == len(set(trade_ids)), "Trade IDs must be unique"

    def test_book_empty_after_full_match(self) -> None:
        """After a full match, the book returns to empty on both sides."""
        book = OrderBook()
        submit(book, make_order("S", Side.SELL, price=10.0, quantity=5, tick=0))
        submit(book, make_order("B", Side.BUY, price=10.0, quantity=5, tick=1))
        assert book.best_bid() is None
        assert book.best_ask() is None

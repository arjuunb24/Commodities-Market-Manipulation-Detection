"""
tests/conftest.py
==================
Shared pytest fixtures for the Veridex test suite.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.core.order_book import BookState, OrderBook, Side


@pytest.fixture
def fresh_book() -> OrderBook:
    """Return a fresh empty OrderBook."""
    return OrderBook()


@pytest.fixture
def rng_42() -> np.random.Generator:
    """Seeded numpy Generator for deterministic test cases."""
    return np.random.default_rng(42)


@pytest.fixture
def normal_state() -> BookState:
    """Typical book state with mid-price at 100."""
    return BookState(
        best_bid=99.0,
        best_ask=101.0,
        mid_price=100.0,
        bid_depth=[(99.0, 20), (98.0, 30)],
        ask_depth=[(101.0, 20), (102.0, 30)],
        last_trade_price=100.0,
        tick=10,
    )


@pytest.fixture
def empty_state() -> BookState:
    """Empty book state with no prices or history."""
    return BookState(
        best_bid=None,
        best_ask=None,
        mid_price=None,
        bid_depth=[],
        ask_depth=[],
        last_trade_price=None,
        tick=0,
    )

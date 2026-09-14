"""
src/core/agents/momentum_trader.py
====================================
Momentum Trader — trend-following agent.

Represents non-commercial (speculative) participants tracking price trends.
Corresponds to CFTC CoT "non-commercial" category (~45% of positions).

Mechanism:
  Computes the N-tick price change (momentum). If the signed change exceeds
  a threshold, places a limit order in the direction of the trend.
  Order size scales linearly with momentum strength (stronger trend → larger order).

This agent is intentionally exploitable by the PumpDump persona, which
relies on momentum traders amplifying a coordinated price burst.
"""

from __future__ import annotations

import uuid
from collections import deque

import numpy as np

from src.core.agents.base import Agent
from src.core.order_book import BookState, Order, Side


class MomentumTrader(Agent):
    """
    Trend-following agent: buys in uptrends, sells in downtrends.

    Tracks a rolling window of last_trade_prices to compute momentum.
    Acts only when |momentum| >= threshold and activation Bernoulli fires.

    Args:
        lookback_ticks: Number of past trade prices to use for trend computation.
        momentum_threshold: Minimum |price change| / ref_price to trigger action.
        order_size_base: Base order quantity.
        order_size_scale: Multiplier: quantity = base + scale * |momentum| * 1000
        arrival_rate_lambda: Per-tick activation probability.
    """

    def __init__(
        self,
        trader_id: str,
        rng: np.random.Generator,
        *,
        lookback_ticks: int = 20,
        momentum_threshold: float = 0.002,
        order_size_base: int = 30,
        order_size_scale: float = 5.0,
        arrival_rate_lambda: float = 0.3,
    ) -> None:
        super().__init__(trader_id, rng)
        self.lookback_ticks = lookback_ticks
        self.momentum_threshold = momentum_threshold
        self.order_size_base = order_size_base
        self.order_size_scale = order_size_scale
        self.arrival_rate_lambda = arrival_rate_lambda

        # Rolling price history — fed by the simulation loop via update_price()
        self._price_history: deque[float] = deque(maxlen=lookback_ticks)

    def update_price(self, price: float) -> None:
        """
        Feed a new trade price into the rolling window.
        Called by the simulation loop after each tick that produced a trade.
        """
        self._price_history.append(price)

    def decide(self, book_state: BookState, tick: int) -> Order | None:
        # --- Activation ---
        if self.rng.random() > self.arrival_rate_lambda:
            return None

        # --- Need enough history ---
        if len(self._price_history) < 2:
            return None

        # --- Compute momentum ---
        oldest = self._price_history[0]
        newest = self._price_history[-1]

        if oldest <= 0:
            return None

        momentum = (newest - oldest) / oldest  # relative price change over window

        if abs(momentum) < self.momentum_threshold:
            return None  # trend not strong enough

        # --- Reference price for order placement ---
        ref_price = book_state.mid_price or book_state.last_trade_price or newest
        if ref_price is None or ref_price <= 0:
            return None

        # --- Order direction follows trend ---
        if momentum > 0:
            side = Side.BUY
            price = ref_price + self.rng.uniform(0, 0.5)  # slightly aggressive
        else:
            side = Side.SELL
            price = ref_price - self.rng.uniform(0, 0.5)

        price = round(max(price, 0.01), 4)

        # Quantity scales with momentum strength
        quantity = int(self.order_size_base + self.order_size_scale * abs(momentum) * 1000)
        quantity = max(1, quantity)

        return Order(
            order_id=str(uuid.uuid4()),
            trader_id=self.trader_id,
            side=side,
            price=price,
            quantity=quantity,
            tick=tick,
            order_type="limit",
        )

"""
src/core/agents/noise_trader.py
================================
Noise Trader — Poisson-arrival random order agent.

Represents uninformed retail participants (non-reportable CoT category).
Order arrival follows a Poisson process (per tick activation probability λ).
Order size and price offset are drawn from uniform distributions.

Role in the simulation:
  - Provides baseline liquidity and prevents the book from freezing
  - Creates "noise" in order flow that provides cover for manipulation agents
  - Population: 20 agents (largest group, ~25% uninformed per CFTC CoT ratios)

Design principle:
  - All randomness from self.rng (seeded independently per agent)
  - No lookahead, no fundamentals view, no trend following
"""

from __future__ import annotations

import uuid

import numpy as np

from src.core.agents.base import Agent
from src.core.order_book import BookState, Order, Side


class NoiseTrader(Agent):
    """
    Uninformed trader that places random small limit orders near mid-price.

    Activation: Bernoulli(lambda_activation) per tick — simulates a Poisson
    process with mean inter-arrival time of 1/λ ticks.

    Order placement:
      - Side: uniform random (buy or sell)
      - Price: mid_price ± Uniform(0, price_offset_ticks) price units
      - Quantity: Uniform(order_size_min, order_size_max)

    When book is empty (no mid-price), the trader uses the last known trade
    price. If no trades have happened yet, the agent does not act this tick.
    """

    def __init__(
        self,
        trader_id: str,
        rng: np.random.Generator,
        *,
        arrival_rate_lambda: float = 0.6,
        order_size_min: int = 1,
        order_size_max: int = 50,
        price_offset_ticks: float = 2.0,
    ) -> None:
        super().__init__(trader_id, rng)
        self.arrival_rate_lambda = arrival_rate_lambda
        self.order_size_min = order_size_min
        self.order_size_max = order_size_max
        self.price_offset_ticks = price_offset_ticks

    def decide(self, book_state: BookState, tick: int) -> Order | None:
        # --- Poisson activation (Bernoulli per tick) ---
        if self.rng.random() > self.arrival_rate_lambda:
            return None  # not active this tick

        # --- Determine reference price ---
        ref_price = book_state.mid_price or book_state.last_trade_price
        if ref_price is None:
            return None  # no price information yet; wait

        # --- Choose side and price ---
        side = Side.BUY if self.rng.random() < 0.5 else Side.SELL
        offset = self.rng.uniform(0, self.price_offset_ticks)

        if side == Side.BUY:
            # Random limit price around mid: can be passive or aggressive
            price = ref_price + self.rng.uniform(-self.price_offset_ticks, self.price_offset_ticks)
        else:
            price = ref_price + self.rng.uniform(-self.price_offset_ticks, self.price_offset_ticks)

        price = round(max(price, 0.01), 4)  # floor at 0.01 (price must be > 0)
        quantity = int(self.rng.integers(self.order_size_min, self.order_size_max + 1))

        return Order(
            order_id=str(uuid.uuid4()),
            trader_id=self.trader_id,
            side=side,
            price=price,
            quantity=quantity,
            tick=tick,
            order_type="limit",
        )

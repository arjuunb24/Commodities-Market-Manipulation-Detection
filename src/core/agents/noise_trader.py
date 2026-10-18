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
  - Uninformed by default: direction is a fair coin. `imbalance_sensitivity`
    can tilt it toward the visible book imbalance, but any value that makes the
    whole retail crowd chase imbalance creates a self-reinforcing loop (buying
    thins the asks → book looks bid-heavy → more buying) that inflated 30-second
    volatility ~10x in testing, so it is off. Instead orders are centred on the
    microprice (bounded imbalance adjustment, see decide()), and the market
    maker leans its quotes on imbalance (see market_maker.py).
  - Unfilled orders are cancelled after a geometric lifetime (mean_order_ttl).
  - Leans against its own position (`position_limit`): a random walk of fills
    otherwise left each account holding 1,000-3,000 contracts by the end of a
    session, which real retail risk limits do not allow.
"""

from __future__ import annotations

import numpy as np

from src.core.agents.base import Agent
from src.core.order_book import BookState, Order, Side


class NoiseTrader(Agent):
    """
    Uninformed trader that places small limit orders around mid-price.

    Activation: Bernoulli(lambda) per tick (discretised Poisson arrivals).

    Order placement:
      - Side: Bernoulli(0.5 + imbalance_sensitivity * book imbalance), default 0.5
      - Price: mid ± Uniform(0, price_offset_ticks) price units (so roughly half
        the orders are marketable and half rest passively)
      - Quantity: log-normal (right-skewed, like real retail order sizes),
        clipped to [order_size_min, order_size_max]

    When there is no mid-price the trader uses the last trade price; if there
    is no price information at all it does not act.
    """

    mean_order_ttl = 40.0

    def __init__(
        self,
        trader_id: str,
        rng: np.random.Generator,
        *,
        arrival_rate_lambda: float = 0.6,
        order_size_min: int = 1,
        order_size_max: int = 50,
        price_offset_ticks: float = 2.0,
        imbalance_sensitivity: float = 0.0,
        microprice_weight: float = 1.5,
        position_limit: int = 2000,
    ) -> None:
        super().__init__(trader_id, rng)
        self.arrival_rate_lambda = arrival_rate_lambda
        self.order_size_min = order_size_min
        self.order_size_max = order_size_max
        self.price_offset_ticks = price_offset_ticks
        self.imbalance_sensitivity = imbalance_sensitivity
        self.microprice_weight = microprice_weight
        self.position_limit = max(1, int(position_limit))
        self.net_position = 0

    def update_inventory(self, side: Side, quantity: int) -> None:
        self.net_position += quantity if side == Side.BUY else -quantity

    def decide(self, book_state: BookState, tick: int) -> Order | None:
        if self.rng.random() > self.arrival_rate_lambda:
            return None

        ref_price = book_state.mid_price or book_state.last_trade_price
        if ref_price is None:
            return None

        lean = 0.35 * float(np.clip(self.net_position / self.position_limit, -1.0, 1.0))
        p_buy = float(np.clip(0.5 + self.imbalance_sensitivity * book_state.imbalance() - lean, 0.15, 0.85))
        side = Side.BUY if self.rng.random() < p_buy else Side.SELL
        # Price around the microprice: mid shifted toward the heavier side of the
        # visible book by at most ~half the spread (Stoikov 2018). Bounded, so it
        # cannot run away, but it lets displayed depth (real or spoofed) nudge
        # where uninformed flow trades.
        if book_state.best_bid is not None and book_state.best_ask is not None:
            half = (book_state.best_ask - book_state.best_bid) / 2 + self.tick_size
            ref_price = ref_price + self.microprice_weight * book_state.imbalance() * half
        price = ref_price + self.rng.uniform(-self.price_offset_ticks, self.price_offset_ticks)
        price = max(price, self.tick_size)

        median_size = max(1.0, (self.order_size_min + self.order_size_max) / 4)
        quantity = int(np.clip(round(self.rng.lognormal(np.log(median_size), 0.7)),
                               self.order_size_min, self.order_size_max))
        return self.limit(side, price, quantity, tick)

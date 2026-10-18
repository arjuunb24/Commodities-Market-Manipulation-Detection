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

Orders are marketable (priced through the opposite best quote) because trend
followers demand immediacy, and each trader respects a position limit, so
herding is bounded the way risk limits bound real CTAs.

The trigger is volatility-scaled: a move counts as a trend when it exceeds
`vol_k` x the typical move over the lookback (a slow average of recent moves),
with `momentum_threshold` as a floor. A fixed 0.1-0.2% threshold meant gold's
trend followers placed about 15 orders a session (gold rarely moves that much in
20 seconds) while natural gas's traded constantly.

This agent is intentionally exploitable by the PumpDump persona, which
relies on momentum traders amplifying a coordinated price burst.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from src.core.agents.base import Agent
from src.core.order_book import BookState, Order, Side


class MomentumTrader(Agent):
    """
    Trend-following agent: buys in uptrends, sells in downtrends.

    Tracks a rolling window of trade prices to compute momentum and acts only
    when |momentum| >= threshold and the activation Bernoulli fires.

    Args:
        lookback_ticks: Number of past per-tick price samples used for the trend.
        momentum_threshold: Minimum |relative price change| to trigger action.
        order_size_base: Base order quantity.
        order_size_scale: quantity = base + scale * |momentum| * 1000
        arrival_rate_lambda: Per-tick activation probability.
        max_position: Absolute position limit (risk limit).
    """

    mean_order_ttl = 8.0

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
        max_position: int = 400,
        vol_k: float = 1.5,
    ) -> None:
        super().__init__(trader_id, rng)
        self.lookback_ticks = lookback_ticks
        self.momentum_threshold = momentum_threshold
        self.order_size_base = order_size_base
        self.order_size_scale = order_size_scale
        self.arrival_rate_lambda = arrival_rate_lambda
        self.max_position = max_position
        self.net_position = 0
        self._price_history: deque[float] = deque(maxlen=lookback_ticks)
        self.vol_k = float(vol_k)
        self._typical_move: float | None = None   # slow average of |lookback move|

    def update_price(self, price: float) -> None:
        """Feed the latest price into the rolling window (the simulation calls this once per tick)."""
        self._price_history.append(price)
        if len(self._price_history) == self._price_history.maxlen and self._price_history[0] > 0:
            move = abs(self._price_history[-1] - self._price_history[0]) / self._price_history[0]
            self._typical_move = move if self._typical_move is None else 0.995 * self._typical_move + 0.005 * move

    def update_inventory(self, side: Side, quantity: int) -> None:
        self.net_position += quantity if side == Side.BUY else -quantity

    def decide(self, book_state: BookState, tick: int) -> Order | None:
        if self.rng.random() > self.arrival_rate_lambda:
            return None
        if len(self._price_history) < 2:
            return None

        oldest = self._price_history[0]
        newest = self._price_history[-1]
        if oldest <= 0:
            return None
        momentum = (newest - oldest) / oldest
        trigger = max(self.momentum_threshold, self.vol_k * (self._typical_move or 0.0))
        if self._typical_move is None or abs(momentum) < trigger:
            return None

        ref_price = book_state.mid_price or book_state.last_trade_price or newest
        if ref_price is None or ref_price <= 0:
            return None

        slip = self.rng.uniform(0.0, 0.0005)
        if momentum > 0:
            if self.net_position >= self.max_position:
                return None
            side = Side.BUY
            price = (book_state.best_ask or ref_price) * (1 + slip)
        else:
            if self.net_position <= -self.max_position:
                return None
            side = Side.SELL
            price = (book_state.best_bid or ref_price) * (1 - slip)

        quantity = int(self.order_size_base + self.order_size_scale * abs(momentum) * 1000)
        return self.limit(side, max(price, self.tick_size), max(1, quantity), tick)

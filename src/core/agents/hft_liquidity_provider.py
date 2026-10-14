"""
src/core/agents/hft_liquidity_provider.py
=========================================
HFT liquidity provider — a legitimate, high-cancel quoting firm.

Real futures markets contain several electronic market makers besides the
designated one. They refresh quotes many times a second, often as a ladder of
several price levels, and cancel the overwhelming majority of their orders
(order-to-trade ratios of 10-100 are normal). Surveillance therefore cannot
flag "cancels a lot", "rests orders on several levels" or "rarely fills" on its
own; it has to find what is specific to manipulation (cancelling one side while
filling on the other, one-sided fake depth). Without these participants every
account that cancels heavily is an outlier and spoofing/layering are trivially
separable.

Mechanism (per tick, with probability `arrival_rate_lambda`):
  1. Cancel all of its still-resting quotes (cancel-and-replace).
  2. Quote both sides, starting at the touch or one tick behind it, as a
     ladder of 1-3 levels per side with sizes of 0.2-0.6x the average visible
     level depth (small enough not to dampen price discovery).
  3. Inventory control (cap = `inventory_cap_depths` x level depth, so it scales
     with the quote size): beyond half the cap it quotes only the
     reducing side (a one-sided ladder, like layering but without a resting
     opposite order); occasionally it crosses the spread with a small order
     to flatten.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from src.core.agents.base import Agent
from src.core.order_book import BookState, Order, Side


class HFTLiquidityProvider(Agent):
    def __init__(
        self,
        trader_id: str,
        rng: np.random.Generator,
        *,
        arrival_rate_lambda: float = 0.8,
        ladder_prob: float = 0.5,
        inventory_cap_depths: float = 6.0,
        flatten_prob: float = 0.05,
    ) -> None:
        super().__init__(trader_id, rng)
        self.arrival_rate_lambda = arrival_rate_lambda
        self.ladder_prob = ladder_prob
        self.inventory_cap_depths = inventory_cap_depths
        self.flatten_prob = flatten_prob
        self.net_inventory = 0
        self._resting: list[str] = []
        self._depth_ref: float | None = None
        self._is_resting: Callable[[str], bool] = lambda _oid: True

    def bind_book(self, is_resting: Callable[[str], bool]) -> None:
        self._is_resting = is_resting

    def update_inventory(self, side: Side, quantity: int) -> None:
        self.net_inventory += quantity if side == Side.BUY else -quantity

    def decide(self, book_state: BookState, tick: int) -> Order | None:
        orders = self.decide_batch(book_state, tick)
        return orders[0] if orders else None

    def _cancel(self, oid: str, tick: int) -> Order:
        return Order(order_id=oid, trader_id=self.trader_id, side=Side.BUY, price=1.0,
                     quantity=1, tick=tick, order_type="cancel")

    def decide_batch(self, state: BookState, tick: int) -> list[Order]:
        if self.rng.random() > self.arrival_rate_lambda:
            return []
        bid, ask = state.best_bid, state.best_ask
        if bid is None or ask is None or ask <= bid:
            return []
        # Slow average of the visible level depth (sizes follow the book as it fills in).
        levels = state.bid_depth + state.ask_depth
        avg = sum(q for _, q in levels) / max(1, len(levels)) if levels else 40.0
        self._depth_ref = avg if self._depth_ref is None else 0.98 * self._depth_ref + 0.02 * avg

        orders = [self._cancel(oid, tick) for oid in self._resting if self._is_resting(oid)]
        self._resting = []

        cap = max(50.0, self.inventory_cap_depths * self._depth_ref)   # scales with quote size
        inv = self.net_inventory / cap
        if abs(inv) > 0.5 and self.rng.random() < self.flatten_prob:
            side = Side.SELL if inv > 0 else Side.BUY
            qty = max(5, int(abs(self.net_inventory) * 0.3))
            orders.append(self.limit(side, bid if side == Side.SELL else ask, qty, tick, ttl=2))

        n_levels = int(self.rng.integers(2, 4)) if self.rng.random() < self.ladder_prob else 1
        sides = []
        if inv < 0.5:
            sides.append(Side.BUY)
        if inv > -0.5:
            sides.append(Side.SELL)
        offset = int(self.rng.random() < 0.5)   # join the touch or sit one level behind
        # Levels at least 1 bp apart: on fine-tick contracts (gold) one-tick ladders
        # would form a dense wall at the touch and damp price discovery.
        step = max(self.tick_size, self.snap(bid * 0.0001))
        # Never quote inside 1 level-step of the mid: joining a fine-tick touch would pin
        # the spread at one tick and turn price moves into a low-volatility staircase.
        mid = 0.5 * (bid + ask)
        base_bid, base_ask = min(bid, self.snap(mid - step)), max(ask, self.snap(mid + step))
        for side in sides:
            for k in range(offset, offset + n_levels):
                price = base_bid - k * step if side == Side.BUY else base_ask + k * step
                qty = max(5, int(self._depth_ref * self.rng.uniform(0.2, 0.6)))
                o = self.limit(side, price, qty, tick, ttl=None)
                self._resting.append(o.order_id)
                orders.append(o)
        return orders

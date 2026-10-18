"""
src/core/agents/execution_algo.py
=================================
Institutional execution algorithm — a legitimate trader of large resting orders.

Asset managers, hedgers and producers work large parent orders through
algorithms that rest big child orders at the touch and cancel and re-post them
whenever the market moves away ("chasing"). Their footprint looks like the
crude version of spoofing: large orders, mostly cancelled, often within
seconds. They also hedge: a small trade in the opposite direction now and
then (another leg, a risk rebalance). Real surveillance has to tell these
apart from spoofers, so the simulator needs them too.

Mechanism:
  idle     with probability `start_prob` per tick, start a parent order:
           random side, total 5-20x the average visible level depth.
  working  keep one child limit order at the near touch (or one tick behind
           it), sized from a heavy-tailed 1-8x level depth. Cancel and re-post
           it when it is no longer at its intended price or after a random
           3-15 tick patience. With probability `take_prob`
           cross the spread with a slice to stay on schedule, and with
           `hedge_prob` send a small opposite-side marketable order. The parent
           completes when the total has filled (or after `max_duration`).
  inventory once its net position exceeds `inventory_cap_depths` x level depth,
           the next parent order is in the reducing direction (an asset manager
           does not keep buying in one account forever; positions reached
           10,000-18,000 contracts without this).
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from src.core.agents.base import Agent
from src.core.order_book import BookState, Order, Side


class ExecutionAlgo(Agent):
    def __init__(
        self,
        trader_id: str,
        rng: np.random.Generator,
        *,
        start_prob: float = 1 / 900,
        take_prob: float = 0.04,
        hedge_prob: float = 0.03,
        max_duration: int = 900,
        inventory_cap_depths: float = 15.0,
    ) -> None:
        super().__init__(trader_id, rng)
        self.start_prob = start_prob
        self.take_prob = take_prob
        self.hedge_prob = hedge_prob
        self.max_duration = max_duration
        self.inventory_cap_depths = float(inventory_cap_depths)
        self.net_inventory = 0
        self._depth_ref: float | None = None
        self._is_resting: Callable[[str], bool] = lambda _oid: True
        self._side: Side | None = None
        self._remaining = 0
        self._started = 0
        self._child: tuple[str, float, int] | None = None   # (order_id, price, deadline tick)
        self._inv_at_start = 0

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
        bid, ask = state.best_bid, state.best_ask
        if bid is None or ask is None or ask <= bid:
            return []
        # Slow average of the visible level depth (sizes follow the book as it fills in).
        levels = state.bid_depth + state.ask_depth
        avg = sum(q for _, q in levels) / max(1, len(levels)) if levels else 40.0
        self._depth_ref = avg if self._depth_ref is None else 0.98 * self._depth_ref + 0.02 * avg

        if self._side is None:
            if self.rng.random() < self.start_prob:
                self._side = Side.BUY if self.rng.random() < 0.5 else Side.SELL
                if abs(self.net_inventory) > self.inventory_cap_depths * self._depth_ref:
                    self._side = Side.SELL if self.net_inventory > 0 else Side.BUY
                self._remaining = int(self._depth_ref * self.rng.uniform(5, 20))
                self._started = tick
                self._inv_at_start = self.net_inventory
            return []

        done = abs(self.net_inventory - self._inv_at_start)
        orders: list[Order] = []
        if done >= self._remaining or tick - self._started > self.max_duration:
            if self._child and self._is_resting(self._child[0]):
                orders.append(self._cancel(self._child[0], tick))
            self._side, self._child = None, None
            return orders

        side = self._side
        behind = self.tick_size if self.rng.random() < 0.5 else 0.0
        touch = (bid - behind) if side == Side.BUY else (ask + behind)
        touch = self.snap(touch)
        left = self._remaining - done
        if self._child is not None:
            oid, price, deadline = self._child
            if not self._is_resting(oid):
                self._child = None
            elif abs(price - touch) > self.tick_size * 1.5 or tick >= deadline:
                orders.append(self._cancel(oid, tick))   # chase / patience ran out
                self._child = None
        if self._child is None:
            mult = float(np.clip(self.rng.lognormal(np.log(2.0), 0.6), 1.0, 8.0))
            qty = min(left, max(10, int(self._depth_ref * mult)))
            o = self.limit(side, touch, qty, tick, ttl=None)
            self._child = (o.order_id, o.price, tick + int(self.rng.integers(3, 16)))
            orders.append(o)
        if self.rng.random() < self.take_prob:
            far = ask if side == Side.BUY else bid
            orders.append(self.limit(side, far, max(5, int(left * 0.1)), tick, ttl=2))
        if self.rng.random() < self.hedge_prob:
            opp = Side.SELL if side == Side.BUY else Side.BUY
            orders.append(self.limit(opp, bid if opp == Side.SELL else ask,
                                     int(self.rng.integers(5, 30)), tick, ttl=2))
        return orders

"""
src/core/agents/base.py
=======================
Abstract base class for all Veridex simulation agents.

Besides the decide() contract, the base class owns three pieces of shared
market plumbing that every realistic participant needs:

  * Deterministic order ids  (<id_prefix><trader_id>-<n>) — no uuid4, so runs
    with the same seed are byte-identical (SRS FR-0.4).
  * Tick-size snapping       — every price lands on the instrument's tick grid.
  * Order time-to-live (TTL) — resting orders that are not filled are cancelled
    after a sampled lifetime. Real limit orders are overwhelmingly cancelled,
    not filled; without this, cancellation is something only manipulators do,
    which makes `cancel_rate` an unrealistically perfect spoofing signal.
"""

from __future__ import annotations

import heapq
from abc import ABC, abstractmethod
from typing import Callable

import numpy as np

from src.core.order_book import BookState, Order, Side, round_to_tick


class Agent(ABC):
    """
    Abstract base class for all Veridex simulation agents.

    Agents are pure decision-makers: they receive a read-only BookState snapshot
    and return an Order instruction (or None). They do NOT submit orders directly
    to the book — the simulation loop collects agent decisions and dispatches them.

    Args:
        trader_id: Unique string identifier for this agent instance.
        rng: An independent seeded numpy.random.Generator (never the global RNG).
    """

    #: Set by the simulation per run (run/commodity scoped) to keep ids unique.
    id_prefix: str = ""
    #: Instrument tick size; the simulation sets the commodity's real tick size.
    tick_size: float = 0.01
    #: Mean resting lifetime (ticks) of an unfilled order; None = good-till-cancel.
    mean_order_ttl: float | None = None

    def __init__(self, trader_id: str, rng: np.random.Generator) -> None:
        self.trader_id = trader_id
        self.rng = rng
        self._order_seq = 0
        self._expiry_heap: list[tuple[int, int, str]] = []

    @abstractmethod
    def decide(self, book_state: BookState, tick: int) -> Order | None:
        """
        Decide what (if anything) to do this tick.

        Must be a pure function of (book_state, tick, self.rng, own state).
        Returning None is always valid. Returned orders must have price > 0
        and quantity > 0, and the agent must handle an empty book gracefully.
        """
        ...

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def next_order_id(self, tag: str = "") -> str:
        """
        Deterministic id. `tag` is a simulator-side audit marker (e.g. "s" = spoof
        leg) used only by effectiveness accounting in src/adversarial; the detector
        never parses order ids.
        """
        self._order_seq += 1
        return f"{self.id_prefix}{self.trader_id}-{self._order_seq}{tag}"

    def snap(self, price: float) -> float:
        return round_to_tick(price, self.tick_size)

    def limit(self, side: Side, price: float, quantity: int, tick: int,
              ttl: float | None = None, exact_ttl: int | None = None, tag: str = "") -> Order:
        """
        Build a limit order on the tick grid and schedule its expiry.

        ttl:       mean of a geometric lifetime (defaults to mean_order_ttl).
        exact_ttl: deterministic lifetime in ticks (used by personas whose cancel
                   timing is itself a strategy parameter).
        """
        order = Order(
            order_id=self.next_order_id(tag),
            trader_id=self.trader_id,
            side=side,
            price=self.snap(price),
            quantity=max(1, int(quantity)),
            tick=tick,
            order_type="limit",
        )
        if exact_ttl is not None:
            heapq.heappush(self._expiry_heap, (tick + max(1, exact_ttl), self._order_seq, order.order_id))
            return order
        mean_ttl = self.mean_order_ttl if ttl is None else ttl
        if mean_ttl is not None and mean_ttl > 0:
            life = int(self.rng.geometric(1.0 / mean_ttl))
            heapq.heappush(self._expiry_heap, (tick + life, self._order_seq, order.order_id))
        return order

    def expire_orders(self, tick: int, is_resting: Callable[[str], bool]) -> list[Order]:
        """Cancel instructions for this agent's orders whose lifetime has elapsed."""
        cancels: list[Order] = []
        while self._expiry_heap and self._expiry_heap[0][0] <= tick:
            _, _, oid = heapq.heappop(self._expiry_heap)
            if is_resting(oid):
                cancels.append(Order(
                    order_id=oid, trader_id=self.trader_id, side=Side.BUY,
                    price=1.0, quantity=1, tick=tick, order_type="cancel",
                ))
        return cancels

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(trader_id={self.trader_id!r})"

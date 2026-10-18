"""
src/core/agents/manipulation/common.py
======================================
Shared building blocks for manipulation personas.

Realism principle: a manipulator is an ordinary market participant who
*sometimes* manipulates. Every manipulator account therefore trades like a
small discretionary trader outside its manipulation window (background
activity). Without this, an account that only exists while manipulating would
let a detector key on "new account appears" rather than on the behaviour
itself, and trader-level ground truth would be trivially separable.

Cover trading (`cover_rate`): real manipulators kept trading genuinely while
they manipulated (Sarao and Coscia ran ordinary flow next to their spoof
orders), so the manipulative orders were a minority of the account's activity.
With cover_rate > 0 every account keeps sending small genuine orders inside the
window too, passive at its own touch or marketable, leaning against its
position so it stays flat. `cover_size` sets their size as a multiple of the
average visible level depth (0 = small retail-size orders). The detector's
cancel/fill/large-order ratios are volume-weighted, so only cover volume
comparable to the scheme's own volume dilutes them. Cost: inventory risk, and
genuine resting depth that offsets the fake depth on the spoofed side.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable

import numpy as np

from src.core.agents.base import Agent
from src.core.agents.occasional_trader import discretionary_order
from src.core.order_book import BookState, Order, Side


class ManipulatorAccount(Agent):
    """
    One trading account controlled by a manipulation persona.

    decide() produces the account's *background* (legitimate-looking) flow:
    occasional small limit orders near the mid that expire like any other
    participant's orders. Persona controllers place the manipulative orders
    directly through limit().
    """

    mean_order_ttl = 40.0

    def __init__(self, trader_id: str, rng: np.random.Generator, *, background_rate: float = 0.05) -> None:
        super().__init__(trader_id, rng)
        self.background_rate = background_rate
        self.net_position = 0
        self._cover_depth_ref: float | None = None

    def update_inventory(self, side: Side, quantity: int) -> None:
        self.net_position += quantity if side == Side.BUY else -quantity

    def decide(self, book_state: BookState, tick: int) -> Order | None:
        # Same code as the legitimate OccasionalTrader, so background flow is no fingerprint.
        return discretionary_order(self, book_state, tick, self.background_rate)

    def cover_order(self, book_state: BookState, tick: int, size_mult: float = 0.0) -> Order | None:
        """One genuine order inside the manipulation window (audit tag "v")."""
        bid, ask = book_state.best_bid, book_state.best_ask
        if bid is None or ask is None or bid <= 0 or ask <= bid:
            return None
        if size_mult > 0:
            # Depth reference frozen at the first cover order: large resting cover
            # orders deepen the book, and sizing off the live depth would feed back.
            if self._cover_depth_ref is None:
                levels = book_state.bid_depth + book_state.ask_depth
                self._cover_depth_ref = sum(q for _, q in levels) / max(1, len(levels)) or 50.0
            qty = max(5, int(self._cover_depth_ref * size_mult * self.rng.uniform(0.5, 1.5)))
        else:
            qty = int(self.rng.integers(5, 40))
        p_buy = 0.5 - 0.4 * float(np.clip(self.net_position / (10.0 * qty + 150.0), -1, 1))
        side = Side.BUY if self.rng.random() < p_buy else Side.SELL
        if self.rng.random() < 0.35:   # take liquidity: trade at the opposite touch
            return self.limit(side, ask if side == Side.BUY else bid, qty, tick, ttl=3, tag="v")
        return self.limit(side, bid if side == Side.BUY else ask, qty, tick, tag="v")


BAND = 0.002   # same ±0.2% near-touch band the book uses for its imbalance signal


def fake_depth_share(state: BookState, side: Side, legs: list[tuple[float, int]]) -> float:
    """
    Share of the visible near-touch depth on `side` that the given (price, qty) legs
    make up at placement time: q_in_band / (existing band volume + q_in_band).
    This is the lever spoofing/layering pull — fake displayed depth — measured
    directly (realised price impact per cycle is too noisy to judge a mutation).
    """
    mid = state.mid_price
    if not mid:
        return 0.0
    q = sum(qty for price, qty in legs if abs(price - mid) / mid <= BAND)
    existing = state.bid_band_volume if side == Side.BUY else state.ask_band_volume
    return q / (existing + q) if (existing + q) > 0 else 0.0


class ManipulationController(ABC):
    """
    Coordinates one or more ManipulatorAccounts for a single episode.

    The simulation loop calls, every tick:
      - expire_orders()  → cancel instructions (processed before new orders)
      - decide_batch()   → new orders (background flow outside the window)
    """

    PERSONA: str = ""

    def __init__(self, accounts: list[ManipulatorAccount], start_tick: int, end_tick: int,
                 busy_period_bias: float = 0.0) -> None:
        self.accounts = accounts
        self.agents: list[Agent] = list(accounts)
        self.start_tick = start_tick
        self.end_tick = end_tick
        # Design assumption (no documented case): act mainly when the market is busy, so the
        # manipulation hides among elevated, naturally explainable activity.
        # 0 = ignore market activity; 1 = never act in quiet periods.
        self.busy_period_bias = float(busy_period_bias)
        # Per-account chance per tick of a genuine cover order inside the window (see module doc).
        self.cover_rate = 0.0
        self.cover_size = 0.0   # cover order size x average level depth (0 = small orders)

    def in_window(self, tick: int) -> bool:
        return self.start_tick <= tick <= self.end_tick

    def decide_batch(self, state: BookState, tick: int) -> list[Order]:
        if self.in_window(tick):
            cover = self._cover_orders(state, tick)
            if self.busy_period_bias > 0 and state.baseline_volume > 0 and not state.is_busy():
                rng = getattr(self, "rng", None) or self.accounts[0].rng
                if rng.random() < self.busy_period_bias:
                    return cover   # quiet market: wait for cover
            return self.manipulate(state, tick) + cover
        orders = []
        for acct in self.accounts:
            o = acct.decide(state, tick)
            if o is not None:
                orders.append(o)
        return orders

    def _cover_orders(self, state: BookState, tick: int) -> list[Order]:
        if self.cover_rate <= 0:
            return []
        orders = []
        for acct in self.accounts:
            if acct.rng.random() < self.cover_rate:
                o = acct.cover_order(state, tick, self.cover_size)
                if o is not None:
                    orders.append(o)
        return orders

    @abstractmethod
    def manipulate(self, state: BookState, tick: int) -> list[Order]:
        """Persona-specific behaviour inside the manipulation window."""

    @abstractmethod
    def parameters_snapshot(self) -> dict:
        """Exact parameters active for this episode (ground-truth audit trail)."""

    def ground_truth_end(self) -> int:
        return self.end_tick

    def expire_orders(self, tick: int, is_resting: Callable[[str], bool]) -> list[Order]:
        cancels: list[Order] = []
        for acct in self.accounts:
            cancels.extend(acct.expire_orders(tick, is_resting))
        return cancels

    def log_activity(self, *_args) -> dict:
        """manipulation_events row (event_id is assigned by the simulation)."""
        return {
            "event_id": "",
            "persona": self.PERSONA,
            "start_tick": self.start_tick,
            "end_tick": self.ground_truth_end(),
            "trader_ids": [a.trader_id for a in self.accounts],
            "parameters": {**self.parameters_snapshot(), "cover_rate": self.cover_rate, "cover_size": self.cover_size},
        }

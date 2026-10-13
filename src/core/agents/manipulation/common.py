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
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable

import numpy as np

from src.core.agents.base import Agent
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

    def update_inventory(self, side: Side, quantity: int) -> None:
        self.net_position += quantity if side == Side.BUY else -quantity

    def decide(self, book_state: BookState, tick: int) -> Order | None:
        if self.rng.random() > self.background_rate:
            return None
        ref = book_state.mid_price or book_state.last_trade_price
        if ref is None or ref <= 0:
            return None
        # Lean against any accumulated position so background flow stays flat.
        p_buy = 0.5 - 0.3 * float(np.clip(self.net_position / 200.0, -1, 1))
        side = Side.BUY if self.rng.random() < p_buy else Side.SELL
        price = ref * (1 + self.rng.uniform(-0.002, 0.002))
        qty = int(self.rng.integers(5, 40))
        return self.limit(side, price, qty, tick)


class ManipulationController(ABC):
    """
    Coordinates one or more ManipulatorAccounts for a single episode.

    The simulation loop calls, every tick:
      - expire_orders()  → cancel instructions (processed before new orders)
      - decide_batch()   → new orders (background flow outside the window)
    """

    PERSONA: str = ""

    def __init__(self, accounts: list[ManipulatorAccount], start_tick: int, end_tick: int) -> None:
        self.accounts = accounts
        self.agents: list[Agent] = list(accounts)
        self.start_tick = start_tick
        self.end_tick = end_tick

    def in_window(self, tick: int) -> bool:
        return self.start_tick <= tick <= self.end_tick

    def decide_batch(self, state: BookState, tick: int) -> list[Order]:
        if self.in_window(tick):
            return self.manipulate(state, tick)
        orders = []
        for acct in self.accounts:
            o = acct.decide(state, tick)
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
            "parameters": self.parameters_snapshot(),
        }

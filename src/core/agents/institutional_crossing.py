"""
Occasional fair-price crosses between independent institutional mandates (wash look-alike).

A broker or asset manager matching unrelated clients' rebalancing orders against
each other inside the spread is legitimate (beneficial ownership changes), yet its
footprint is a wash ring's: the same few accounts trading mostly with each other,
near the mid, often netting out over a session (a roll or a rebalance in both
directions). Surveillance clears it by ownership, not by the order flow.

Two regimes:
  * a low steady rate of small crosses (`cycle_rate`), as before;
  * crossing SESSIONS (start chance `session_prob` per tick, 150-700 ticks): a burst
    at a rate and lot size drawn per session from ranges that reach the wash
    persona's (it crosses 120-lot trades at up to 0.5 per tick). A session either
    transfers inventory from the longer to the shorter account (nets out), or
    moves it one way with probability `directional_bias` (a real allocation).
Outside crossing activity the client accounts trade like occasional traders
(the manipulators' background code), so they are not idle shells.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from src.core.agents.base import Agent
from src.core.agents.occasional_trader import discretionary_order
from src.core.order_book import BookState, Order, Side

BACKGROUND_RATE = 0.03


class InstitutionalCrossingDesk(Agent):
    """Model a broker matching unrelated client rebalancing orders at fair value."""

    def __init__(
        self,
        trader_id: str,
        rng: np.random.Generator,
        *,
        n_accounts: int | None = None,
        cycle_rate: float = 0.01,
        order_size_min: int = 10,
        order_size_max: int = 80,
        session_prob: float = 1 / 2500,
        background_rate: float = BACKGROUND_RATE,
    ) -> None:
        super().__init__(trader_id, rng)
        n_accounts = int(n_accounts) if n_accounts else int(rng.integers(2, 6))
        n_accounts = max(2, n_accounts)
        self.accounts = [
            AgentAccount(f"{trader_id}_account_{i}", account_rng)
            for i, account_rng in enumerate(rng.spawn(n_accounts))
        ]
        self.cycle_rate = float(np.clip(cycle_rate, 0.0, 1.0))
        self.order_size_min = max(1, int(order_size_min))
        self.order_size_max = max(self.order_size_min, int(order_size_max))
        self.session_prob = float(session_prob)
        self.background_rate = float(background_rate)
        self.directional_bias = float(rng.uniform(0.0, 0.6))
        self._pair_index = 0
        self._session: dict | None = None

    def decide(self, book_state: BookState, tick: int) -> Order | None:
        return None

    def _start_session(self, tick: int) -> None:
        r = self.rng
        lot = float(r.uniform(40, 200))
        self._session = {
            "end": tick + int(r.integers(150, 700)),
            "rate": float(r.uniform(0.05, 0.4)),
            "lot": (max(5, int(lot * 0.6)), int(lot * 1.4)),
            # one-way transfer (allocation) or netting rebalance
            "from": int(r.integers(0, len(self.accounts))) if r.random() < self.directional_bias else None,
        }

    def decide_batch(self, state: BookState, tick: int) -> list[Order]:
        if state.best_bid is None or state.best_ask is None or state.best_ask <= state.best_bid:
            return []
        if self._session is not None and tick > self._session["end"]:
            self._session = None
        if self._session is None and self.rng.random() < self.session_prob:
            self._start_session(tick)

        s = self._session
        rate = s["rate"] if s is not None else self.cycle_rate
        if self.rng.random() >= rate:
            orders = []
            for a in self.accounts:   # not crossing this tick: ordinary occasional flow
                o = discretionary_order(a, state, tick, self.background_rate)
                if o is not None:
                    orders.append(o)
            return orders

        a = self.accounts[self._pair_index % len(self.accounts)]
        b = self.accounts[(self._pair_index + 1) % len(self.accounts)]
        self._pair_index = (self._pair_index + 1) % len(self.accounts)

        tick_size = a.tick_size
        low, high = state.best_bid + tick_size, state.best_ask - tick_size
        if low > high:
            return []
        price = float(np.clip(a.snap((state.best_bid + state.best_ask) / 2), low, high))
        lo, hi = s["lot"] if s is not None else (self.order_size_min, self.order_size_max)
        quantity = int(np.clip(self.rng.lognormal(np.log((lo + hi) / 2), 0.45), lo, hi))

        if s is not None and s["from"] is not None:
            # Allocation: the source account sells, whoever it is paired with buys.
            src = self.accounts[s["from"]]
            a_side = Side.SELL if a is src else (Side.BUY if b is src else
                                                 (Side.BUY if self.rng.random() < 0.5 else Side.SELL))
        elif a.net_position == b.net_position:
            # Rebalancing tends to transfer inventory from the more-long account to
            # the more-short one; ties are resolved randomly, not by a fixed side.
            a_side = Side.BUY if self.rng.random() < 0.5 else Side.SELL
        else:
            a_side = Side.SELL if a.net_position > b.net_position else Side.BUY
        b_side = Side.SELL if a_side == Side.BUY else Side.BUY
        return [
            a.limit(a_side, price, quantity, tick, exact_ttl=1),
            b.limit(b_side, price, quantity, tick, exact_ttl=1),
        ]

    def expire_orders(
        self, tick: int, is_resting: Callable[[str], bool]
    ) -> list[Order]:
        return [
            order
            for account in self.accounts
            for order in account.expire_orders(tick, is_resting)
        ]


class AgentAccount(Agent):
    """Client account owned by an institutional crossing desk."""

    mean_order_ttl = 40.0

    def __init__(self, trader_id: str, rng: np.random.Generator) -> None:
        super().__init__(trader_id, rng)
        self.net_position = 0

    def decide(self, book_state: BookState, tick: int) -> Order | None:
        return None

    def update_inventory(self, side: Side, quantity: int) -> None:
        self.net_position += quantity if side == Side.BUY else -quantity

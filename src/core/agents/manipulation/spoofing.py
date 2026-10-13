"""
src/core/agents/manipulation/spoofing.py
=========================================
Spoofing persona — bid or offer with the intent to cancel before execution.

CFTC Definition (Commodity Exchange Act §4c(a)(5)(C)):
  "Bidding or offering with the intent to cancel the bid or offer before execution."

How real spoofing makes money (Coscia 2015, Sarao 2015, see
docs/enforcement_case_notes.md) — and how this agent reproduces it:

  1. Place a SMALL genuine order on one side at the touch
     (e.g. a sell at the best ask) — the order the spoofer actually wants filled.
  2. Place a LARGE non-bona-fide order on the OTHER side, a few ticks behind the
     best price (e.g. a big bid just under the best bid). It is far enough back
     to be unlikely to trade, but close enough to sit in the visible depth.
  3. Other participants read the fake depth as buying pressure (in this ABM the
     market maker leans its quotes and noise traders tilt to buying, see
     BookState.imbalance()), the price ticks up and the genuine sell is lifted.
  4. Cancel the large order after `cancellation_delay_ticks`; cancel the genuine
     order too if it has not filled. Repeat, sometimes in the other direction.

The behavioural footprint this leaves is trader-level, not market-level:
large orders with short lifetimes that are almost always cancelled, on the side
OPPOSITE to where the same account's executions happen.

Evasion knobs (each documented in docs/enforcement_case_notes.md):
  size_jitter     ±random spoof size (CFTC 2020 advisory #1, Coscia round 2)
  cancel_jitter   random cancel delay instead of a fixed one (advisory #2)
  n_split_orders  spoof split into several smaller orders on consecutive ticks
                  behind the touch (Coscia "split into batches", Tower layering)
  n_accounts      >1 → the genuine order and the spoof come from DIFFERENT
                  accounts (JPMorgan cross-trader spoofing, advisory #3)

Order ids carry an audit tag ("s" spoof leg, "g" genuine leg) used only for
effectiveness accounting, never by the detector.
"""

from __future__ import annotations

import numpy as np

from src.core.agents.manipulation.common import ManipulationController, ManipulatorAccount
from src.core.order_book import BookState, Order, Side


class SpoofingAgent(ManipulationController):
    """
    Spoofing persona (one or more accounts).

    Args:
        order_size_multiplier: Total spoof size as a multiple of average visible level depth.
        cancellation_delay_ticks: Mean ticks the spoof rests before it is cancelled.
        frequency: Probability per tick of starting a new spoof cycle (when idle).
        price_aggressiveness: Distance behind the same-side best price (fraction).
        size_jitter: Spoof size is multiplied by U(1 - j, 1 + j).
        cancel_jitter: Cancel delay drawn from delay * U(1 - j, 1 + j).
        n_split_orders: Number of orders the spoof is split into (consecutive ticks).
        n_accounts: Accounts used; with >= 2, genuine and spoof legs use different accounts.
        spoof_side: "buy", "sell" or "both" (direction chosen per cycle).
    """

    PERSONA = "spoofing"

    def __init__(
        self,
        trader_id: str,
        rng: np.random.Generator,
        *,
        order_size_multiplier: float = 8.0,
        cancellation_delay_ticks: int = 5,
        frequency: float = 0.4,
        price_aggressiveness: float = 0.0005,
        size_jitter: float = 0.0,
        cancel_jitter: float = 0.0,
        n_split_orders: int = 1,
        n_accounts: int = 1,
        start_tick: int = 0,
        end_tick: int = 10000,
        spoof_side: str = "both",
        background_rate: float = 0.05,
    ) -> None:
        n_accounts = max(1, int(n_accounts))
        acct_rngs = rng.spawn(n_accounts)
        names = [trader_id] + [f"{trader_id}_{k}" for k in range(1, n_accounts)]
        accounts = [ManipulatorAccount(n, r, background_rate=background_rate) for n, r in zip(names, acct_rngs)]
        super().__init__(accounts, start_tick, end_tick)
        self.trader_id = trader_id
        self.rng = rng
        self.order_size_multiplier = order_size_multiplier
        self.cancellation_delay_ticks = max(1, int(cancellation_delay_ticks))
        self.frequency = frequency
        self.price_aggressiveness = price_aggressiveness
        self.size_jitter = float(np.clip(size_jitter, 0.0, 0.9))
        self.cancel_jitter = float(np.clip(cancel_jitter, 0.0, 0.9))
        self.n_split_orders = max(1, int(n_split_orders))
        self.spoof_side = spoof_side
        self._cycle_busy_until = -1

    @property
    def account(self) -> ManipulatorAccount:
        return self.accounts[0]

    def parameters_snapshot(self) -> dict:
        return {
            "order_size_multiplier": self.order_size_multiplier,
            "cancellation_delay_ticks": self.cancellation_delay_ticks,
            "frequency": self.frequency,
            "price_aggressiveness": self.price_aggressiveness,
            "size_jitter": self.size_jitter,
            "cancel_jitter": self.cancel_jitter,
            "n_split_orders": self.n_split_orders,
            "n_accounts": len(self.accounts),
            "spoof_side": self.spoof_side,
        }

    def _pick_side(self) -> Side:
        if self.spoof_side == "buy":
            return Side.BUY
        if self.spoof_side == "sell":
            return Side.SELL
        return Side.BUY if self.rng.random() < 0.5 else Side.SELL

    def manipulate(self, state: BookState, tick: int) -> list[Order]:
        if tick < self._cycle_busy_until:
            return []  # current spoof is still resting
        if self.rng.random() > self.frequency:
            return []
        if state.best_bid is None or state.best_ask is None:
            return []

        # Account roles: genuine leg and spoof leg come from different accounts when available.
        idx = self.rng.permutation(len(self.accounts))
        genuine_acct = self.accounts[idx[0]]
        spoof_acct = self.accounts[idx[1]] if len(self.accounts) > 1 else genuine_acct
        tick_size = spoof_acct.tick_size
        spoof_side = self._pick_side()

        levels = state.bid_depth + state.ask_depth
        avg_level = sum(q for _, q in levels) / max(1, len(levels))
        jitter = self.rng.uniform(1 - self.size_jitter, 1 + self.size_jitter)
        total_qty = max(int(avg_level * self.order_size_multiplier * jitter), 50)
        delay = max(1, int(round(self.cancellation_delay_ticks
                                 * self.rng.uniform(1 - self.cancel_jitter, 1 + self.cancel_jitter))))
        genuine_qty = int(self.rng.integers(5, 30))

        if spoof_side == Side.BUY:
            base = min(state.best_bid * (1 - self.price_aggressiveness), state.best_ask - tick_size)
            step = -tick_size
            genuine = genuine_acct.limit(Side.SELL, state.best_ask, genuine_qty, tick, exact_ttl=delay + 2, tag="g")
        else:
            base = max(state.best_ask * (1 + self.price_aggressiveness), state.best_bid + tick_size)
            step = tick_size
            genuine = genuine_acct.limit(Side.BUY, state.best_bid, genuine_qty, tick, exact_ttl=delay + 2, tag="g")

        orders = [genuine]
        k = self.n_split_orders
        weights = self.rng.dirichlet(np.ones(k) * 3.0) if k > 1 else np.array([1.0])
        for i in range(k):
            qty = max(1, int(round(total_qty * weights[i])))
            orders.append(spoof_acct.limit(spoof_side, base + i * step, qty, tick, exact_ttl=delay, tag="s"))
        self._cycle_busy_until = tick + delay
        return orders

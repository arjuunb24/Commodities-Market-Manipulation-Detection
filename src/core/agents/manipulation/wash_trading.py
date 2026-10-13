"""
src/core/agents/manipulation/wash_trading.py
=============================================
Wash trading persona — pre-arranged crosses between colluding accounts.

CFTC Definition:
  "Entering into, or purporting to enter into, transactions to give the
  appearance of purchases and sales but resulting in no net change in
  the trader's market position."

How it works in practice (Kraft 2015 and the wash-trade cases in
docs/enforcement_case_notes.md) — and how this persona reproduces it:

  * Two accounts under common control trade with EACH OTHER. One rests a limit
    order and the other immediately takes it at the same price and size. In a
    real book this is done inside the spread so no third party gets in between.
  * The roles alternate, so each account's net position stays ≈ 0 while gross
    volume is large — the trades print volume, nothing changes hands.
  * Any leg that could not be matched with the partner is cancelled straight
    away (the colluders do not want genuine exposure).

The previous implementation had each account independently send aggressive
orders at mid ± 0.1%, which mostly traded against *other* participants — i.e.
two directional traders, not a wash. That has been replaced by the cross above.

Evasion knobs (docs/enforcement_case_notes.md):
  ring_size        accounts per colluding group; crosses rotate A→B→C→A so no
                   single pair dominates (Coinbase cyclical washes)
  n_groups         independent groups; the persona's total cross rate is SPLIT
                   across them (account proliferation dilutes per-account
                   signals without adding volume — advisory #3, Kraft round 2)
  size_jitter      0 = fixed scripted lot size; higher = varied sizes
  camouflage_rate  share of cycles in which an account instead takes a small
                   genuine directional trade from the market (Kraft round 2,
                   advisory #4) — costs real exposure and wash volume
price_deviation_from_mid only moves the cross price inside the spread, so it
has almost no market effect and is not offered to the strategist.
"""

from __future__ import annotations

import numpy as np

from src.core.agents.manipulation.common import ManipulationController, ManipulatorAccount
from src.core.order_book import BookState, Order, Side


WASH_BASE_LOT = 120


class WashTradingRing(ManipulationController):
    """One group of colluding accounts crossing pre-arranged trades with each other."""

    PERSONA = "wash_trading"

    def __init__(
        self,
        accounts: list[ManipulatorAccount],
        rng: np.random.Generator,
        *,
        trade_frequency: float,
        price_deviation_from_mid: float,
        start_tick: int,
        end_tick: int,
        size_jitter: float = 0.0,
        camouflage_rate: float = 0.0,
        n_groups: int = 1,
    ) -> None:
        super().__init__(accounts, start_tick, end_tick)
        self.rng = rng
        self.trade_frequency = trade_frequency
        self.price_deviation_from_mid = price_deviation_from_mid
        self.size_jitter = float(np.clip(size_jitter, 0.0, 0.95))
        self.camouflage_rate = float(np.clip(camouflage_rate, 0.0, 0.9))
        self.n_groups = max(1, int(n_groups))
        self._next = 0

    @classmethod
    def create(
        cls,
        rng: np.random.Generator,
        pair_index: int,
        *,
        trade_frequency: float = 0.5,
        price_deviation_from_mid: float = 0.001,
        start_tick: int = 0,
        end_tick: int = 10000,
        background_rate: float = 0.05,
        ring_size: int = 2,
        size_jitter: float = 0.0,
        camouflage_rate: float = 0.0,
        n_groups: int = 1,
    ) -> "WashTradingRing":
        ring_size = max(2, int(ring_size))
        ctrl_rng, *acct_rngs = rng.spawn(ring_size + 1)
        accounts = [
            ManipulatorAccount(f"wash_{pair_index}{chr(ord('a') + k)}", acct_rngs[k], background_rate=background_rate)
            for k in range(ring_size)
        ]
        return cls(
            accounts, ctrl_rng,
            trade_frequency=trade_frequency,
            price_deviation_from_mid=price_deviation_from_mid,
            start_tick=start_tick, end_tick=end_tick,
            size_jitter=size_jitter, camouflage_rate=camouflage_rate, n_groups=n_groups,
        )

    @property
    def buyer(self) -> ManipulatorAccount:  # kept for backward compatibility
        return self.accounts[0]

    @property
    def seller(self) -> ManipulatorAccount:
        return self.accounts[1]

    def parameters_snapshot(self) -> dict:
        return {
            "trade_frequency": self.trade_frequency * self.n_groups,
            "price_deviation_from_mid": self.price_deviation_from_mid,
            "n_groups": self.n_groups,
            "ring_size": len(self.accounts),
            "size_jitter": self.size_jitter,
            "camouflage_rate": self.camouflage_rate,
        }

    def _lot(self) -> int:
        j = self.size_jitter
        return max(5, int(round(WASH_BASE_LOT * self.rng.uniform(1 - j, 1 + j))))

    def manipulate(self, state: BookState, tick: int) -> list[Order]:
        if self.rng.random() > self.trade_frequency:
            return []
        if state.best_bid is None or state.best_ask is None:
            return []

        n = len(self.accounts)
        a = self.accounts[self._next % n]
        b = self.accounts[(self._next + 1) % n]
        self._next += 1

        if self.rng.random() < self.camouflage_rate:
            # Genuine directional trade against the market, sized like normal flow.
            side = Side.BUY if a.net_position <= 0 else Side.SELL
            price = state.best_ask if side == Side.BUY else state.best_bid
            return [a.limit(side, price, int(self.rng.integers(10, 40)), tick, exact_ttl=2, tag="c")]

        tick_size = a.tick_size
        lo, hi = state.best_bid + tick_size, state.best_ask - tick_size
        if hi < lo - 1e-12:
            return []  # no price strictly inside the spread: wait rather than trade with strangers
        mid = (state.best_bid + state.best_ask) / 2
        price = float(np.clip(mid * (1 + self.price_deviation_from_mid * self.rng.uniform(-1, 1)), lo, hi))
        qty = self._lot()
        rest_side = Side.SELL if self.rng.random() < 0.5 else Side.BUY
        take_side = Side.BUY if rest_side == Side.SELL else Side.SELL
        # Unmatched remainders are pulled on the next tick.
        leg1 = a.limit(rest_side, price, qty, tick, exact_ttl=1, tag="w")
        leg2 = b.limit(take_side, leg1.price, qty, tick, exact_ttl=1, tag="w")
        return [leg1, leg2]


# Backward-compatible name (a pair is a ring of size 2).
WashTradingPair = WashTradingRing

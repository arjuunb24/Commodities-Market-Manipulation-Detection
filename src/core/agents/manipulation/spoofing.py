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
  reprice_to_touch  1 → while resting, the spoof is cancelled and re-placed so it
                  stays the same distance behind the moving best price
                  (Sarao 2015: orders automatically modified to avoid execution
                  while keeping the fake depth visible near the touch)
  flip_after_cancel probability of "flipping" when the spoof is pulled: trading
                  aggressively in the genuine order's direction straight away
                  (Oystacher 2015)
  busy_period_bias  act mainly when the market is busy (advisory #5)

Order ids carry an audit tag ("s" spoof leg, "g" genuine leg) used only for
effectiveness accounting, never by the detector.
"""

from __future__ import annotations

import numpy as np

from src.core.agents.manipulation.common import ManipulationController, ManipulatorAccount, fake_depth_share
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
        reprice_to_touch: 1 = keep the resting spoof pinned behind the moving best price.
        flip_after_cancel: Probability of an aggressive trade in the genuine direction when the spoof is pulled.
        busy_period_bias: Probability of skipping a tick when the market is quieter than usual.
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
        reprice_to_touch: int = 0,
        flip_after_cancel: float = 0.0,
        busy_period_bias: float = 0.0,
        start_tick: int = 0,
        end_tick: int = 10000,
        spoof_side: str = "both",
        background_rate: float = 0.05,
    ) -> None:
        n_accounts = max(1, int(n_accounts))
        acct_rngs = rng.spawn(n_accounts)
        names = [trader_id] + [f"{trader_id}_{k}" for k in range(1, n_accounts)]
        accounts = [ManipulatorAccount(n, r, background_rate=background_rate) for n, r in zip(names, acct_rngs)]
        super().__init__(accounts, start_tick, end_tick, busy_period_bias=busy_period_bias)
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
        self.reprice_to_touch = int(bool(reprice_to_touch))
        self.flip_after_cancel = float(np.clip(flip_after_cancel, 0.0, 1.0))
        self._cycle_busy_until = -1
        # Resting spoof legs of the current cycle: [order_id, account, side, qty, price, cancel_at, level]
        self._legs: list[list] = []
        self._pending_flip: tuple | None = None   # (account, side, qty) to fire when the spoof is pulled
        self._depth_shares: list[float] = []

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
            "reprice_to_touch": self.reprice_to_touch,
            "flip_after_cancel": self.flip_after_cancel,
            "busy_period_bias": self.busy_period_bias,
            "spoof_side": self.spoof_side,
            "realised_fake_depth_share": round(float(np.mean(self._depth_shares)), 4) if self._depth_shares else 0.0,
        }

    def _pick_side(self) -> Side:
        if self.spoof_side == "buy":
            return Side.BUY
        if self.spoof_side == "sell":
            return Side.SELL
        return Side.BUY if self.rng.random() < 0.5 else Side.SELL

    def _target_price(self, state: BookState, side: Side, level: int, tick_size: float) -> float:
        if side == Side.BUY:
            base = min(state.best_bid * (1 - self.price_aggressiveness), state.best_ask - tick_size)
            return base - level * tick_size
        base = max(state.best_ask * (1 + self.price_aggressiveness), state.best_bid + tick_size)
        return base + level * tick_size

    def _reprice(self, state: BookState, tick: int) -> list[Order]:
        """Sarao-style: cancel and re-place resting spoof legs that drifted from their target distance."""
        orders: list[Order] = []
        for leg in self._legs:
            oid, acct, side, qty, price, cancel_at, level = leg
            if cancel_at <= tick:
                continue
            target = acct.snap(self._target_price(state, side, level, acct.tick_size))
            if abs(target - price) < acct.tick_size / 2:
                continue
            orders.append(Order(order_id=oid, trader_id=acct.trader_id, side=side, price=1.0,
                                quantity=1, tick=tick, order_type="cancel"))
            new = acct.limit(side, target, qty, tick, exact_ttl=max(1, cancel_at - tick), tag="s")
            orders.append(new)
            leg[0], leg[4] = new.order_id, new.price
        return orders

    def manipulate(self, state: BookState, tick: int) -> list[Order]:
        if state.best_bid is None or state.best_ask is None:
            return []
        if tick < self._cycle_busy_until:
            # The current spoof is still resting.
            return self._reprice(state, tick) if self.reprice_to_touch else []

        orders: list[Order] = []
        if self._pending_flip is not None:
            # Oystacher-style flip: the spoof was just pulled — trade aggressively in the genuine direction.
            acct, side, qty = self._pending_flip
            price = state.best_bid if side == Side.SELL else state.best_ask
            orders.append(acct.limit(side, price, qty, tick, exact_ttl=2, tag="f"))
            self._pending_flip = None
        self._legs = []

        if self.rng.random() > self.frequency:
            return orders

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
        genuine_side = Side.SELL if spoof_side == Side.BUY else Side.BUY
        genuine_price = state.best_ask if genuine_side == Side.SELL else state.best_bid
        orders.append(genuine_acct.limit(genuine_side, genuine_price, genuine_qty, tick, exact_ttl=delay + 2, tag="g"))

        k = self.n_split_orders
        weights = self.rng.dirichlet(np.ones(k) * 3.0) if k > 1 else np.array([1.0])
        for i in range(k):
            qty = max(1, int(round(total_qty * weights[i])))
            o = spoof_acct.limit(spoof_side, self._target_price(state, spoof_side, i, tick_size), qty, tick,
                                 exact_ttl=delay, tag="s")
            orders.append(o)
            self._legs.append([o.order_id, spoof_acct, spoof_side, qty, o.price, tick + delay, i])
        self._depth_shares.append(fake_depth_share(state, spoof_side, [(leg[4], leg[3]) for leg in self._legs]))
        if self.flip_after_cancel > 0 and self.rng.random() < self.flip_after_cancel:
            self._pending_flip = (genuine_acct, genuine_side, int(genuine_qty * 2))
        self._cycle_busy_until = tick + delay
        return orders

"""
src/core/agents/lookalikes.py
=============================
Legitimate participants whose footprint overlaps a manipulation persona's.

Real surveillance spends most of its time on false positives: behaviour that looks
like manipulation but has a business reason. The simulator's other legitimate
agents never came near the personas' features (run seed333: wash rows had
mutual_cpty_share >= 0.55 while the benign 99th percentile was 0.26; layering
orders were >= 9.9x the median size, benign p99 7.0x), so the detector separated
the classes by sheer distance and scored ~0.95 F1 whatever the personas did.
Each agent below reproduces one documented legitimate practice that real
surveillance desks routinely have to clear:

  QuoteProtectingLP   (spoofing look-alike) a non-designated liquidity provider
                      quoting LARGE size near the touch, sometimes axed to one side
                      or laddered over a few levels. When one side is hit it pulls
                      the other side at once (adverse-selection protection), so its
                      large orders are cancelled while it is filled on the opposite
                      side - the spoof_signature pattern, for a legitimate reason.
  LadderHedger        (layering look-alike) a producer/consumer working a hedge
                      as a one-sided ladder of large orders on consecutive levels,
                      placed in one tick and re-placed whenever the market moves
                      away (cancel-and-replace). Unlike layering it genuinely wants
                      those orders filled, and only rarely trades the other side.
  FundFamily          (pump-and-dump look-alike) an asset manager allocating one
                      decision over several sub-accounts: passive accumulation, then
                      urgent marketable buying when behind schedule, a hold, and an
                      exit. Coordinated, same-direction, cancel-light, net-long then
                      net-short - the same footprint, without the price ignition the
                      exit depends on.
(The wash look-alike, a broker crossing client orders, is
institutional_crossing.InstitutionalCrossingDesk.)

Every instance draws its own style from wide ranges (sizes, rates, patience), so
the population overlaps the personas at its tail rather than by construction.
All accounts trade like occasional traders when not working (same code as the
manipulators' background flow), so no background fingerprint separates them.
"""

from __future__ import annotations

from collections import deque
from typing import Callable

import numpy as np

from src.core.agents.base import Agent, DepthReference
from src.core.agents.occasional_trader import discretionary_order
from src.core.order_book import BookState, Order, Side

BACKGROUND_RATE = 0.03


def _cancel(agent: Agent, oid: str, tick: int) -> Order:
    return Order(order_id=oid, trader_id=agent.trader_id, side=Side.BUY, price=1.0,
                 quantity=1, tick=tick, order_type="cancel")


class _DepthTracking(Agent):
    """Shared plumbing: slow level-depth reference, inventory, book binding, background flow."""

    mean_order_ttl = 40.0

    def __init__(self, trader_id: str, rng: np.random.Generator) -> None:
        super().__init__(trader_id, rng)
        self.net_position = 0
        self._depth = DepthReference()
        self._is_resting: Callable[[str], bool] = lambda _oid: True

    def bind_book(self, is_resting: Callable[[str], bool]) -> None:
        self._is_resting = is_resting

    def update_inventory(self, side: Side, quantity: int) -> None:
        self.net_position += quantity if side == Side.BUY else -quantity

    def _update_depth(self, state: BookState) -> float:
        return self._depth.update(state)   # frozen after a warm-up (base.DepthReference)

    def decide(self, book_state: BookState, tick: int) -> Order | None:
        orders = self.decide_batch(book_state, tick)
        return orders[0] if orders else None

    def decide_batch(self, state: BookState, tick: int) -> list[Order]:  # pragma: no cover - overridden
        return []


class QuoteProtectingLP(_DepthTracking):
    """Large-size liquidity provider that pulls the opposite quote when one side is hit."""

    def __init__(self, trader_id: str, rng: np.random.Generator, *,
                 session_prob: float = 1 / 1500) -> None:
        super().__init__(trader_id, rng)
        r = self.rng
        self.session_prob = float(session_prob)
        self.size_range = (float(r.uniform(1.5, 4.0)), float(r.uniform(5.0, 12.0)))   # x level depth
        self.axed_prob = float(r.uniform(0.2, 0.6))       # sessions quoting size on one side only
        self.ladder_prob = float(r.uniform(0.1, 0.5))
        self.refresh_rate = float(r.uniform(0.15, 0.5))
        self.patience = (int(r.integers(3, 8)), int(r.integers(10, 40)))
        self.behind_max = int(r.integers(0, 3))           # ticks behind the touch
        self._session_end = -1
        self._axe: Side | None = None
        self._quotes: list[tuple[str, Side]] = []
        self._quote_mid = 0.0
        self._deadline = 0
        self._pause_until = -1
        self._last_position = 0

    def decide_batch(self, state: BookState, tick: int) -> list[Order]:
        bid, ask = state.best_bid, state.best_ask
        if bid is None or ask is None or ask <= bid:
            return []
        depth = self._update_depth(state)
        if tick > self._session_end:
            if self._quotes:
                return self._pull(tick)
            if self.rng.random() < self.session_prob:
                self._session_end = tick + int(self.rng.integers(300, 1500))
                self._axe = (Side.BUY if self.rng.random() < 0.5 else Side.SELL) \
                    if self.rng.random() < self.axed_prob else None
            else:
                o = discretionary_order(self, state, tick, BACKGROUND_RATE)
                return [o] if o is not None else []

        orders: list[Order] = []
        filled = self.net_position != self._last_position
        hit_side = (Side.BUY if self.net_position > self._last_position else Side.SELL) if filled else None
        self._last_position = self.net_position
        self._quotes = [(oid, s) for oid, s in self._quotes if self._is_resting(oid)]
        if hit_side is not None and self._quotes:
            # Protection: one side traded, pull everything (the opposite side first) and wait.
            orders += self._pull(tick, first=Side.SELL if hit_side == Side.BUY else Side.BUY)
            self._pause_until = tick + int(self.rng.integers(1, 8))
            return orders
        mid = 0.5 * (bid + ask)
        if self._quotes and (tick >= self._deadline or abs(mid - self._quote_mid) > 2 * self.tick_size):
            return self._pull(tick)   # stale: the market moved or patience ran out
        if self._quotes or tick < self._pause_until or self.rng.random() > self.refresh_rate:
            return orders
        # Inventory control: a large position makes the reducing side the axe.
        cap = 8.0 * depth
        axe = self._axe
        if abs(self.net_position) > cap:
            axe = Side.SELL if self.net_position > 0 else Side.BUY
        behind = int(self.rng.integers(0, self.behind_max + 1)) * self.tick_size
        n_levels = int(self.rng.integers(2, 4)) if self.rng.random() < self.ladder_prob else 1
        for side in (Side.BUY, Side.SELL):
            big = axe is None or side == axe
            lo, hi = self.size_range if big else (0.3, 1.0)
            if not big and self.rng.random() < 0.3:
                continue   # axed: sometimes no quote at all on the other side
            top = (bid - behind) if side == Side.BUY else (ask + behind)
            for k in range(n_levels if big else 1):
                price = top - k * self.tick_size if side == Side.BUY else top + k * self.tick_size
                qty = max(5, int(depth * self.rng.uniform(lo, hi)))
                o = self.limit(side, price, qty, tick, ttl=None)
                self._quotes.append((o.order_id, side))
                orders.append(o)
        self._quote_mid = mid
        self._deadline = tick + int(self.rng.integers(*self.patience))
        return orders

    def _pull(self, tick: int, first: Side | None = None) -> list[Order]:
        quotes = sorted(self._quotes, key=lambda q: q[1] != first) if first else self._quotes
        out = [_cancel(self, oid, tick) for oid, _ in quotes if self._is_resting(oid)]
        self._quotes = []
        return out


class LadderHedger(_DepthTracking):
    """Commercial hedger working a large one-sided program as a re-placed ladder."""

    def __init__(self, trader_id: str, rng: np.random.Generator, *,
                 start_prob: float = 1 / 2000) -> None:
        super().__init__(trader_id, rng)
        r = self.rng
        self.start_prob = float(start_prob)
        self.n_levels = (2, int(r.integers(3, 6)))
        self.level_size = (float(r.uniform(0.8, 2.0)), float(r.uniform(2.5, 6.0)))   # x level depth
        self.spacing_ticks = int(r.integers(1, 4))
        self.chase_ticks = int(r.integers(2, 6))
        self.patience = (int(r.integers(8, 20)), int(r.integers(25, 90)))
        self.hedge_prob = float(r.uniform(0.005, 0.05))    # small opposite-side adjustments (other legs, rolls)
        self.behind_max = int(r.integers(1, 5))            # scale orders resting a few ticks back
        self._side: Side | None = None
        self._remaining = 0
        self._started = 0
        self._inv_start = 0
        self._ladder: list[str] = []
        self._ladder_px = 0.0
        self._deadline = 0
        self._pause_until = -1

    def decide_batch(self, state: BookState, tick: int) -> list[Order]:
        bid, ask = state.best_bid, state.best_ask
        if bid is None or ask is None or ask <= bid:
            return []
        depth = self._update_depth(state)
        if self._side is None:
            if self.rng.random() < self.start_prob:
                self._side = Side.BUY if self.rng.random() < 0.5 else Side.SELL
                self._remaining = int(depth * self.rng.uniform(10, 40))
                self._started, self._inv_start = tick, self.net_position
                return []
            o = discretionary_order(self, state, tick, BACKGROUND_RATE)
            return [o] if o is not None else []

        side = self._side
        done = abs(self.net_position - self._inv_start)
        self._ladder = [oid for oid in self._ladder if self._is_resting(oid)]
        if done >= self._remaining or tick - self._started > 2500:
            out = [_cancel(self, oid, tick) for oid in self._ladder]
            self._side, self._ladder = None, []
            return out
        orders: list[Order] = []
        touch = bid if side == Side.BUY else ask
        away = (touch - self._ladder_px) if side == Side.BUY else (self._ladder_px - touch)
        if self._ladder and (tick >= self._deadline or away > self.chase_ticks * self.tick_size):
            orders += [_cancel(self, oid, tick) for oid in self._ladder]   # chase / re-place
            self._ladder = []
            self._pause_until = tick + int(self.rng.integers(0, 4))
        if not self._ladder and tick >= self._pause_until:
            behind = int(self.rng.integers(0, self.behind_max + 1)) * self.tick_size
            top = (bid - behind) if side == Side.BUY else (ask + behind)
            left = self._remaining - done
            for k in range(int(self.rng.integers(self.n_levels[0], self.n_levels[1] + 1))):
                step = k * self.spacing_ticks * self.tick_size
                price = top - step if side == Side.BUY else top + step
                qty = min(left, max(10, int(depth * self.rng.uniform(*self.level_size))))
                o = self.limit(side, price, qty, tick, ttl=None)
                self._ladder.append(o.order_id)
                orders.append(o)
            self._ladder_px = top
            self._deadline = tick + int(self.rng.integers(*self.patience))
        if self.rng.random() < self.hedge_prob:
            opp = Side.SELL if side == Side.BUY else Side.BUY
            orders.append(self.limit(opp, bid if opp == Side.SELL else ask,
                                     int(self.rng.integers(5, 30)), tick, ttl=2))
        return orders


class FundAccount(Agent):
    """One sub-account of a FundFamily (orders are placed by the family)."""

    mean_order_ttl = 40.0

    def __init__(self, trader_id: str, rng: np.random.Generator) -> None:
        super().__init__(trader_id, rng)
        self.net_position = 0

    def update_inventory(self, side: Side, quantity: int) -> None:
        self.net_position += quantity if side == Side.BUY else -quantity

    def decide(self, book_state: BookState, tick: int) -> Order | None:
        return discretionary_order(self, book_state, tick, BACKGROUND_RATE)


class FundFamily(Agent):
    """Asset manager allocating one directional decision over several sub-accounts."""

    def __init__(self, trader_id: str, rng: np.random.Generator, *,
                 n_accounts: int | None = None, start_prob: float = 1 / 4000) -> None:
        super().__init__(trader_id, rng)
        r = self.rng
        n = int(n_accounts) if n_accounts else int(r.integers(3, 9))
        self.accounts = [FundAccount(f"{trader_id}_{i}", a_rng) for i, a_rng in enumerate(r.spawn(n))]
        self.start_prob = float(start_prob)
        self._mids: deque[float] = deque(maxlen=300)
        self._plan: dict | None = None

    def decide(self, book_state: BookState, tick: int) -> Order | None:
        return None

    def expire_orders(self, tick: int, is_resting: Callable[[str], bool]) -> list[Order]:
        return [o for a in self.accounts for o in a.expire_orders(tick, is_resting)]

    def _new_plan(self, tick: int) -> dict:
        r = self.rng
        trend = (self._mids[-1] - self._mids[0]) if len(self._mids) > 1 else 0.0
        side = (Side.BUY if trend >= 0 else Side.SELL) if r.random() < 0.6 and trend != 0 \
            else (Side.BUY if r.random() < 0.5 else Side.SELL)
        entry = int(r.integers(150, 900))
        exit_len = int(r.integers(150, 700))
        hold = int(r.integers(50, 4000))
        stagger = float(r.uniform(0.0, 0.6))
        return {
            "side": side, "start": tick, "entry": entry, "hold": hold, "exit": exit_len,
            "target": {a.trader_id: int(r.integers(40, 600)) for a in self.accounts},
            "offset": {a.trader_id: int(r.integers(0, int(stagger * entry) + 1)) for a in self.accounts},
            "start_pos": {a.trader_id: a.net_position for a in self.accounts},
            # How hard the urgent phase pushes: most plans just take the touch; news-driven
            # ones sweep through it about as hard as a pump's ignition (0.0002-0.005).
            "urgency": float(r.uniform(0.0, 0.0006)) if r.random() < 0.6 else float(r.uniform(0.0006, 0.003)),
            "urgent_rate": float(r.uniform(0.2, 0.6)),
            "urgent_lot": (10, int(r.integers(40, 80))),
        }

    def decide_batch(self, state: BookState, tick: int) -> list[Order]:
        bid, ask = state.best_bid, state.best_ask
        if bid is None or ask is None or ask <= bid:
            return []
        self._mids.append(0.5 * (bid + ask))
        if self._plan is None and self.rng.random() < self.start_prob:
            self._plan = self._new_plan(tick)
        p = self._plan
        orders: list[Order] = []
        finished = True
        for a in self.accounts:
            phase = None
            if p is not None:
                rel = tick - p["start"] - p["offset"][a.trader_id]
                if 0 <= rel < p["entry"]:
                    phase = "passive_in" if rel < 0.7 * p["entry"] else "urgent_in"
                elif p["entry"] + p["hold"] <= rel < p["entry"] + p["hold"] + p["exit"]:
                    phase = "passive_out" if rel < p["entry"] + p["hold"] + 0.6 * p["exit"] else "urgent_out"
                if rel < p["entry"] + p["hold"] + p["exit"]:
                    finished = False
            if phase is None:
                o = a.decide(state, tick)
                if o is not None:
                    orders.append(o)
                continue
            built = (a.net_position - p["start_pos"][a.trader_id]) * (1 if p["side"] == Side.BUY else -1)
            entering = phase.endswith("_in")
            side = p["side"] if entering else (Side.SELL if p["side"] == Side.BUY else Side.BUY)
            left = (p["target"][a.trader_id] - built) if entering else built
            if left <= 0:
                continue
            u = a.rng.random()
            if phase.startswith("passive") and u < 0.25:
                price = bid if side == Side.BUY else ask
                orders.append(a.limit(side, price, min(left, int(a.rng.integers(5, 30))), tick,
                                      exact_ttl=int(a.rng.integers(10, 30))))
            elif phase.startswith("urgent") and u < p["urgent_rate"]:
                k = a.rng.uniform(0.25, 1.0) * p["urgency"]
                price = ask * (1 + k) if side == Side.BUY else bid * (1 - k)
                orders.append(a.limit(side, price, min(left, int(a.rng.integers(*p["urgent_lot"]))), tick, exact_ttl=3))
        if p is not None and finished:
            self._plan = None
        return orders

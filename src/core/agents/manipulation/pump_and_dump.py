"""
src/core/agents/manipulation/pump_and_dump.py
==============================================
Pump-and-dump persona — coordinated momentum ignition followed by distribution.

In futures markets the pump-and-dump pattern shows up as *momentum ignition*
(see docs/enforcement_case_notes.md, and SEC/CFTC descriptions of the tactic):

  1. ACCUMULATE  Several coordinated accounts quietly build a long position with
                 passive bids at the touch (small, patient, little price impact).
  2. IGNITE      The same accounts then fire a burst of aggressive, marketable
                 buys. The price jumps and trend followers (MomentumTrader) pile in.
  3. DISTRIBUTE  While the followers are still buying, the coalition offers its
                 inventory passively at the ask — selling into the demand it created.
  4. DUMP        Whatever is left is sold aggressively; the price falls back.

Parameter semantics (unchanged names so persona_config / the strategist still work):
  burst_duration_ticks    length of ACCUMULATE + IGNITE (ignition = last 30%)
  dump_delay_ticks        length of DISTRIBUTE before the aggressive DUMP starts
  n_coordinated_accounts  number of colluding accounts
  accumulation_size       target position PER ACCOUNT built during ACCUMULATE

Evasion knobs (docs/enforcement_case_notes.md, Nefedova et al.):
  stagger              each account runs the schedule with its own random delay
                       of up to stagger x burst_duration ("staggered accumulation,
                       appearing like independent decisions")
  ignition_aggression  how far through the ask the ignition buys are priced
                       (lower = less impact and less detectable)
  decoy_rate           probability per tick of a small opposite-side trade while
                       accumulating / distributing (interleaving, advisory #4)

Detection footprint: at trader level, accounts that are net buyers (first
passively, then aggressively) and later flip to net sellers, while their own
buying coincides with the window's price rise.
"""

from __future__ import annotations

from enum import Enum

import numpy as np

from src.core.agents.manipulation.common import ManipulationController, ManipulatorAccount
from src.core.order_book import BookState, Order, Side


class PumpPhase(str, Enum):
    ACCUMULATE = "accumulate"
    IGNITE = "ignite"
    DISTRIBUTE = "distribute"
    DUMP = "dump"
    IDLE = "idle"


IGNITION_FRACTION = 0.3
DUMP_DURATION = 160


class PumpDumpCoalition(ManipulationController):
    """Coordinated multi-account momentum-ignition / pump-and-dump persona."""

    PERSONA = "pump_and_dump"

    def __init__(
        self,
        accounts: list[ManipulatorAccount],
        rng: np.random.Generator,
        *,
        start_tick: int,
        end_tick: int,
        burst_duration_ticks: int,
        dump_delay_ticks: int,
        accumulation_size: int,
        stagger: float = 0.0,
        ignition_aggression: float = 0.002,
        decoy_rate: float = 0.0,
    ) -> None:
        super().__init__(accounts, start_tick, end_tick)
        self.rng = rng
        self.burst_duration_ticks = max(10, int(burst_duration_ticks))
        self.dump_delay_ticks = max(1, int(dump_delay_ticks))
        self.accumulation_size = max(10, int(accumulation_size))
        self.stagger = float(np.clip(stagger, 0.0, 1.0))
        self.ignition_aggression = float(max(0.0, ignition_aggression))
        self.decoy_rate = float(np.clip(decoy_rate, 0.0, 0.9))
        max_off = int(self.stagger * self.burst_duration_ticks)
        self.offsets = [int(rng.integers(0, max_off + 1)) if max_off > 0 else 0 for _ in accounts]
        self.phase = PumpPhase.IDLE

    @classmethod
    def create(
        cls,
        rng: np.random.Generator,
        *,
        n_coordinated_accounts: int = 3,
        burst_duration_ticks: int = 200,
        dump_delay_ticks: int = 100,
        accumulation_size: int = 150,
        start_tick: int = 5000,
        end_tick: int = 7000,
        episode_index: int = 0,
        background_rate: float = 0.05,
        stagger: float = 0.0,
        ignition_aggression: float = 0.002,
        decoy_rate: float = 0.0,
    ) -> "PumpDumpCoalition":
        n = max(1, int(n_coordinated_accounts))
        ctrl_rng, *acct_rngs = rng.spawn(n + 1)
        accounts = [
            ManipulatorAccount(f"pnd_{episode_index}_{i}", acct_rngs[i], background_rate=background_rate)
            for i in range(n)
        ]
        return cls(
            accounts, ctrl_rng,
            start_tick=start_tick,
            end_tick=end_tick,
            burst_duration_ticks=burst_duration_ticks,
            dump_delay_ticks=dump_delay_ticks,
            accumulation_size=accumulation_size,
            stagger=stagger,
            ignition_aggression=ignition_aggression,
            decoy_rate=decoy_rate,
        )

    def parameters_snapshot(self) -> dict:
        return {
            "n_coordinated_accounts": len(self.accounts),
            "burst_duration_ticks": self.burst_duration_ticks,
            "dump_delay_ticks": self.dump_delay_ticks,
            "accumulation_size": self.accumulation_size,
            "stagger": self.stagger,
            "ignition_aggression": self.ignition_aggression,
            "decoy_rate": self.decoy_rate,
        }

    def ground_truth_end(self) -> int:
        span = self.burst_duration_ticks + self.dump_delay_ticks + DUMP_DURATION + max(self.offsets, default=0)
        return min(self.end_tick, self.start_tick + span)

    def phase_at(self, tick: int, offset: int = 0) -> PumpPhase:
        rel = tick - self.start_tick - offset
        ignite_at = int(self.burst_duration_ticks * (1 - IGNITION_FRACTION))
        end_rel = self.burst_duration_ticks + self.dump_delay_ticks + DUMP_DURATION
        if not self.in_window(tick) or rel < 0 or rel >= end_rel:
            return PumpPhase.IDLE
        if rel < ignite_at:
            return PumpPhase.ACCUMULATE
        if rel < self.burst_duration_ticks:
            return PumpPhase.IGNITE
        if rel < self.burst_duration_ticks + self.dump_delay_ticks:
            return PumpPhase.DISTRIBUTE
        return PumpPhase.DUMP

    def update_phase(self, tick: int) -> None:
        """Coalition-level phase (schedule of the earliest account)."""
        self.phase = self.phase_at(tick, min(self.offsets, default=0))

    def in_window(self, tick: int) -> bool:
        return self.start_tick <= tick <= self.ground_truth_end()

    def manipulate(self, state: BookState, tick: int) -> list[Order]:
        self.update_phase(tick)
        if state.best_bid is None or state.best_ask is None:
            return []
        orders: list[Order] = []
        target = self.accumulation_size
        for acct, offset in zip(self.accounts, self.offsets):
            r = acct.rng.random()
            pos = acct.net_position
            phase = self.phase_at(tick, offset)
            if phase in (PumpPhase.ACCUMULATE, PumpPhase.DISTRIBUTE) and acct.rng.random() < self.decoy_rate:
                # Decoy: small trade AGAINST the account's current direction.
                side = Side.SELL if phase == PumpPhase.ACCUMULATE else Side.BUY
                price = state.best_bid if side == Side.SELL else state.best_ask
                orders.append(acct.limit(side, price, int(acct.rng.integers(5, 15)), tick, exact_ttl=2, tag="d"))
                continue
            if phase == PumpPhase.IDLE:
                o = acct.decide(state, tick)  # not (yet / any more) in the scheme: trade normally
                if o is not None:
                    orders.append(o)
                continue
            if phase == PumpPhase.ACCUMULATE:
                if r < 0.25 and pos < target:
                    qty = min(int(acct.rng.integers(5, 25)), target - pos)
                    orders.append(acct.limit(Side.BUY, state.best_bid, qty, tick, exact_ttl=20))
            elif phase == PumpPhase.IGNITE:
                if r < 0.6 and pos < 2 * target:
                    price = state.best_ask * (1 + acct.rng.uniform(0.25, 1.0) * self.ignition_aggression)
                    orders.append(acct.limit(Side.BUY, price, int(acct.rng.integers(20, 60)), tick, exact_ttl=3))
            elif phase == PumpPhase.DISTRIBUTE:
                if r < 0.35 and pos > 0:
                    qty = min(int(acct.rng.integers(10, 40)), pos)
                    orders.append(acct.limit(Side.SELL, state.best_ask, qty, tick, exact_ttl=15))
            elif phase == PumpPhase.DUMP:
                if r < 0.6 and pos > 0:
                    qty = min(int(acct.rng.integers(25, 70)), pos)
                    price = state.best_bid * (1 - acct.rng.uniform(0.0005, 0.002))
                    orders.append(acct.limit(Side.SELL, price, qty, tick, exact_ttl=3))
        return orders

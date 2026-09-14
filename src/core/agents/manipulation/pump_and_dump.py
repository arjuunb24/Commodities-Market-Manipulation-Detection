"""
src/core/agents/manipulation/pump_and_dump.py
==============================================
Pump-and-Dump Coalition — multi-account coordinated buying burst
designed to trigger momentum traders, followed by a coordinated dump.

Grounded in:
  - CFTC v. Nefedova et al. (2020): Silver futures coordinated manipulation
  (see docs/enforcement_case_notes.md)

Mechanism:
  Phase 1 (PUMP): N coordinated accounts simultaneously buy aggressively
  over burst_duration_ticks ticks. This creates genuine upward price pressure
  that triggers MomentumTrader agents to amplify the move.

  Phase 2 (DUMP): After dump_delay_ticks ticks, the same accounts sell off
  their accumulated positions, causing a price reversal.

  The key signal for detection: the same accounts that were buying heavily
  reverse to selling at the reversal — detected as a net-position reversal
  pattern combined with the volume concentration feature.

Key parameter distinction from spoofing:
  Pump-and-dump orders are NOT cancelled — they are executed trades that
  create real inventory (then reversed). This is why the manipulation is
  harder to detect purely from cancel_rate; it requires tracking
  net-position changes over the episode window.
"""

from __future__ import annotations

import uuid
from enum import Enum

import numpy as np

from src.core.agents.base import Agent
from src.core.order_book import BookState, Order, Side


class PumpPhase(str, Enum):
    ACCUMULATE = "accumulate"
    HOLD = "hold"
    DUMP = "dump"
    DONE = "done"


class PumpDumpAccount(Agent):
    """
    One account in a pump-and-dump coalition.

    The simulation loop coordinates all accounts in the coalition via
    the shared PumpDumpCoalition controller.

    Args:
        phase: Externally set by PumpDumpCoalition controller each tick.
        accumulation_size: Target position size to build during pump phase.
        dump_order_size: Order size during dump phase.
    """

    def __init__(
        self,
        trader_id: str,
        rng: np.random.Generator,
        coalition_id: str,
        *,
        trade_frequency: float = 0.3,
        price_aggression: float = 0.005,
        start_tick: int = 0,
        dump_delay_ticks: int = 50,
        end_tick: int = 10000,
    ) -> None:
        super().__init__(trader_id, rng)
        self.coalition_id = coalition_id
        self.trade_frequency = trade_frequency
        self.price_aggression = price_aggression
        self.start_tick = start_tick
        
        # When to transition from ACCUMULATE to DUMP
        self.dump_tick = start_tick + dump_delay_ticks
        self.end_tick = end_tick
        
        # Target total inventory to accumulate before stopping
        self.accumulation_size = int(self.rng.integers(50000, 100000))
        self.phase = PumpPhase.HOLD
        self.net_inventory: int = 0

    def set_phase(self, phase: PumpPhase) -> None:
        """Called by PumpDumpCoalition controller to transition phase."""
        self.phase = phase

    def update_inventory(self, side: Side, quantity: int) -> None:
        """Track net inventory for dump-phase sizing."""
        if side == Side.BUY:
            self.net_inventory += quantity
        else:
            self.net_inventory -= quantity

    def decide(self, book_state: BookState, tick: int) -> Order | None:
        if tick < self.start_tick or tick > self.end_tick:
            return None

        ref_price = book_state.mid_price or book_state.last_trade_price
        if ref_price is None or ref_price <= 0:
            return None

        if self.phase == PumpPhase.ACCUMULATE:
            if self.net_inventory >= self.accumulation_size:
                return None  # already fully accumulated

            # Buy aggressively, pushing the price up
            buy_price = round(ref_price * (1 + self.price_aggression), 4)
            buy_qty = min(
                int(self.rng.integers(500, 1000)),
                self.accumulation_size - self.net_inventory,
            )

            return Order(
                order_id=str(uuid.uuid4()),
                trader_id=self.trader_id,
                side=Side.BUY,
                price=buy_price,
                quantity=max(1, buy_qty),
                tick=tick,
                order_type="limit",
            )

        elif self.phase == PumpPhase.DUMP:
            if self.net_inventory <= 0:
                return None  # nothing left to sell

            # Sell aggressively to lock in profits
            sell_price = round(ref_price * (1 - self.price_aggression), 4)
            sell_qty = min(
                int(self.rng.integers(1000, 5000)),
                self.net_inventory,
            )

            return Order(
                order_id=str(uuid.uuid4()),
                trader_id=self.trader_id,
                side=Side.SELL,
                price=sell_price,
                quantity=max(1, sell_qty),
                tick=tick,
                order_type="limit",
            )

        return None  # HOLD or DONE phase


class PumpDumpCoalition:
    """
    Controller that coordinates a group of PumpDumpAccount agents.

    Manages phase transitions:
      - [start_tick, start_tick + burst_duration_ticks): ACCUMULATE
      - [start_tick + burst_duration_ticks, start_tick + dump_delay_ticks): HOLD
      - [start_tick + dump_delay_ticks, end_tick]: DUMP

    Usage:
        coalition = PumpDumpCoalition.create(rng, **params)
        # In simulation loop:
        coalition.update_phase(tick)
    """

    PERSONA = "pump_and_dump"

    def __init__(
        self,
        accounts: list[PumpDumpAccount],
        start_tick: int,
        end_tick: int,
        burst_duration_ticks: int,
        dump_delay_ticks: int,
        parameters: dict,
    ) -> None:
        self.accounts = accounts
        self.start_tick = start_tick
        self.end_tick = end_tick
        self.burst_duration_ticks = burst_duration_ticks
        self.dump_delay_ticks = dump_delay_ticks
        self.parameters = parameters
        self.agents: list[Agent] = accounts

    @classmethod
    def create(
        cls,
        rng: np.random.Generator,
        *,
        n_coordinated_accounts: int = 3,
        burst_duration_ticks: int = 200,
        dump_delay_ticks: int = 300,
        start_tick: int = 5000,
        end_tick: int = 7000,
    ) -> "PumpDumpCoalition":
        child_rngs = rng.spawn(n_coordinated_accounts)
        coalition_id = f"pd_coalition_{str(uuid.uuid4())[:8]}"
        accounts = [
            PumpDumpAccount(
                f"pumpdump_acct_{i}",
                child_rngs[i],
                coalition_id,
                start_tick=start_tick,
                dump_delay_ticks=dump_delay_ticks,
                end_tick=end_tick,
            )
            for i in range(n_coordinated_accounts)
        ]
        params = {
            "n_coordinated_accounts": n_coordinated_accounts,
            "burst_duration_ticks": burst_duration_ticks,
            "dump_delay_ticks": dump_delay_ticks,
        }
        return cls(accounts, start_tick, end_tick, burst_duration_ticks, dump_delay_ticks, params)

    def update_phase(self, tick: int) -> None:
        """Called by simulation loop each tick to update all account phases."""
        if tick < self.start_tick or tick > self.end_tick:
            phase = PumpPhase.HOLD
        elif tick < self.start_tick + self.burst_duration_ticks:
            phase = PumpPhase.ACCUMULATE
        elif tick < self.start_tick + self.dump_delay_ticks:
            phase = PumpPhase.HOLD
        else:
            phase = PumpPhase.DUMP

        for account in self.accounts:
            account.set_phase(phase)

    def log_activity(self) -> dict:
        """Return a manipulation_events row for this coalition."""
        # FIX: The manipulation is only active during ACCUMULATE and DUMP phases.
        # It ends after dump_delay_ticks + a few ticks to offload the inventory.
        # We should NOT label the rest of the episode (which could be thousands of ticks of inactivity).
        actual_end_tick = min(self.end_tick, self.start_tick + self.dump_delay_ticks + 50)
        
        return {
            "event_id": str(uuid.uuid4()),
            "persona": self.PERSONA,
            "start_tick": self.start_tick,
            "end_tick": actual_end_tick,
            "trader_ids": [a.trader_id for a in self.accounts],
            "parameters": self.parameters,
        }

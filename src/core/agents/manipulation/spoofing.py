"""
src/core/agents/manipulation/spoofing.py
=========================================
Spoofing Agent — places large passive orders to shift book imbalance,
then cancels before fill.

CFTC Definition (Commodity Exchange Act §4c(a)(5)):
  "Bidding or offering with the intent to cancel the bid or offer before execution."

Grounded in enforcement cases:
  - Coscia v. CFTC (2014): first criminal spoofing conviction
  - Tower Research Capital (2019): $67.4M settlement for layering/spoofing
  (see docs/enforcement_case_notes.md for full case summaries)

Mechanism:
  1. Agent places a large limit order on the book (far enough from mid to rest,
     close enough to shift the bid-ask imbalance signal visibly).
  2. After cancellation_delay_ticks ticks, the agent cancels the order.
  3. The book imbalance signal (bid volume / ask volume) shifts during the
     window when the spoof order is resting, potentially influencing other
     agents' decisions (particularly MomentumTraders).

Logging:
  Every spoofing episode is logged to manipulation_events with:
  - event_id, persona='spoofing', start_tick, end_tick, trader_ids, parameters
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field

import numpy as np

from src.core.agents.base import Agent
from src.core.order_book import BookState, Order, Side


@dataclass
class SpoofingEvent:
    """Tracks a single pending spoof order (placed, awaiting cancellation)."""
    order_id: str
    placed_tick: int
    cancel_at_tick: int
    side: Side


class SpoofingAgent(Agent):
    """
    Single-account spoofing agent.

    Places large one-sided limit orders to create false book imbalance,
    then cancels them before they can be filled.

    Args:
        order_size_multiplier: Multiplier on average book depth for spoof order size.
        cancellation_delay_ticks: Ticks between spoof placement and cancel instruction.
        frequency: Probability of initiating a new spoof on each active tick.
        price_aggressiveness: Distance from mid as fraction of mid_price to place spoof.
        start_tick: First tick the agent is active.
        end_tick: Last tick the agent is active (inclusive).
        spoof_side: Which side to spoof ('buy' inflates bid, 'sell' inflates ask).
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
        price_aggressiveness: float = 0.003,
        start_tick: int = 0,
        end_tick: int = 10000,
        spoof_side: str = "buy",
    ) -> None:
        super().__init__(trader_id, rng)
        self.order_size_multiplier = order_size_multiplier
        self.cancellation_delay_ticks = cancellation_delay_ticks
        self.frequency = frequency
        self.price_aggressiveness = price_aggressiveness
        self.start_tick = start_tick
        self.end_tick = end_tick
        self.spoof_side = Side.BUY if spoof_side == "buy" else Side.SELL

        # Pending spoof orders awaiting cancellation
        self._pending: list[SpoofingEvent] = []

        # Episode tracking for manipulation_events log
        self._episode_start: int | None = None
        self._events_log: list[dict] = []  # collected by simulation loop

    def parameters_snapshot(self) -> dict:
        """Return a snapshot of current parameters for manipulation_events logging."""
        return {
            "order_size_multiplier": self.order_size_multiplier,
            "cancellation_delay_ticks": self.cancellation_delay_ticks,
            "frequency": self.frequency,
            "price_aggressiveness": self.price_aggressiveness,
            "spoof_side": self.spoof_side.value,
        }

    def decide(self, book_state: BookState, tick: int) -> Order | None:
        """
        On active ticks, occasionally places a large passive spoof order.
        The simulation loop calls get_cancels() separately to retrieve
        cancel instructions for pending spoof orders.
        """
        if tick < self.start_tick or tick > self.end_tick:
            return None

        if self.rng.random() > self.frequency:
            return None

        ref_price = book_state.mid_price or book_state.last_trade_price
        if ref_price is None or ref_price <= 0:
            return None

        # Compute spoof order size: ~8x average visible book depth
        total_depth = sum(qty for _, qty in book_state.bid_depth + book_state.ask_depth)
        avg_depth = total_depth / max(len(book_state.bid_depth) + len(book_state.ask_depth), 1)
        spoof_qty = max(int(avg_depth * self.order_size_multiplier), 100)

        # Spoof price: close to mid but not crossing (won't fill immediately)
        if self.spoof_side == Side.BUY:
            spoof_price = round(ref_price * (1 - self.price_aggressiveness), 4)
        else:
            spoof_price = round(ref_price * (1 + self.price_aggressiveness), 4)

        spoof_price = max(spoof_price, 0.01)
        order_id = str(uuid.uuid4())

        # Register for cancellation
        self._pending.append(SpoofingEvent(
            order_id=order_id,
            placed_tick=tick,
            cancel_at_tick=tick + self.cancellation_delay_ticks,
            side=self.spoof_side,
        ))

        return Order(
            order_id=order_id,
            trader_id=self.trader_id,
            side=self.spoof_side,
            price=spoof_price,
            quantity=spoof_qty,
            tick=tick,
            order_type="limit",
        )

    def get_cancels(self, tick: int) -> list[Order]:
        """
        Return cancel instructions for all spoof orders whose cancellation
        delay has elapsed. Called by the simulation loop each tick.
        """
        due = [e for e in self._pending if e.cancel_at_tick <= tick]
        self._pending = [e for e in self._pending if e.cancel_at_tick > tick]

        cancel_orders = []
        for event in due:
            cancel_orders.append(Order(
                order_id=event.order_id,
                trader_id=self.trader_id,
                side=event.side,
                price=1.0,   # irrelevant for cancel
                quantity=1,  # irrelevant for cancel
                tick=tick,
                order_type="cancel",
            ))
        return cancel_orders

    def log_activity(self, tick: int) -> dict | None:
        """
        Called once per tick by simulation loop. Returns a manipulation_events
        dict when a new episode starts (first active tick of the persona window).
        """
        if tick == self.start_tick:
            return {
                "event_id": str(uuid.uuid4()),
                "persona": self.PERSONA,
                "start_tick": self.start_tick,
                "end_tick": self.end_tick,
                "trader_ids": [self.trader_id],
                "parameters": self.parameters_snapshot(),
            }
        return None

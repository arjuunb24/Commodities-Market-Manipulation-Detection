"""
src/core/agents/manipulation/layering.py
=========================================
Layering Agent — multi-level spoofing across consecutive price levels.

Grounded in:
  - CFTC v. Tower Research Capital (2019): $67.4M settlement for layering
  (see docs/enforcement_case_notes.md)

Mechanism:
  Places N simultaneous limit orders at consecutive price levels on one
  side of the book (all on the same side — all bids or all asks). This
  creates a false impression of large-order depth across multiple levels,
  shifting both the bid-ask imbalance and the visible book depth on that side.

  After cancellation_delay_ticks ticks, all N layer orders are cancelled.

Distinction from simple spoofing:
  - Spoofing: one large order at one price level
  - Layering: N smaller orders across N consecutive price levels
  The Order Size Variance feature is specifically designed to detect
  this difference from normal market-maker quoting behaviour.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass

import numpy as np

from src.core.agents.base import Agent
from src.core.order_book import BookState, Order, Side


@dataclass
class LayerEvent:
    """Tracks one resting layer order awaiting cancellation."""
    order_id: str
    placed_tick: int
    cancel_at_tick: int


class LayeringAgent(Agent):
    """
    Multi-level layering agent.

    Args:
        n_layers: Number of simultaneous layer orders per layering event.
        layer_spacing: Price distance between consecutive layers as fraction of mid.
        cancellation_delay_ticks: Ticks before all layers are cancelled.
        frequency: Probability of initiating a new layering event per active tick.
        layer_side: Side to layer on ('buy' = bid side, 'sell' = ask side).
        layer_size: Quantity per layer order.
        start_tick: First tick of the persona's active window.
        end_tick: Last tick of the active window (inclusive).
    """

    PERSONA = "layering"

    def __init__(
        self,
        trader_id: str,
        rng: np.random.Generator,
        *,
        n_layers: int = 4,
        layer_spacing: float = 0.002,
        cancellation_delay_ticks: int = 3,
        frequency: float = 0.2,
        layer_side: str = "buy",
        layer_size: int = 30,
        start_tick: int = 0,
        end_tick: int = 10000,
    ) -> None:
        super().__init__(trader_id, rng)
        self.n_layers = n_layers
        self.layer_spacing = layer_spacing
        self.cancellation_delay_ticks = cancellation_delay_ticks
        self.frequency = frequency
        self.layer_side = Side.BUY if layer_side == "buy" else Side.SELL
        self.layer_size = layer_size
        self.start_tick = start_tick
        self.end_tick = end_tick

        self._pending: list[LayerEvent] = []

    def parameters_snapshot(self) -> dict:
        return {
            "n_layers": self.n_layers,
            "layer_spacing": self.layer_spacing,
            "cancellation_delay_ticks": self.cancellation_delay_ticks,
            "frequency": self.frequency,
            "layer_side": self.layer_side.value,
            "layer_size": self.layer_size,
        }

    def decide_batch(self, book_state: BookState, tick: int) -> list[Order]:
        """
        Returns 0 or N layer orders for this tick.
        The simulation loop must call this instead of decide() for LayeringAgent.
        Returns an empty list when not activating.
        """
        if tick < self.start_tick or tick > self.end_tick:
            return []

        if self.rng.random() > self.frequency:
            return []

        ref_price = book_state.mid_price or book_state.last_trade_price
        if ref_price is None or ref_price <= 0:
            return []

        orders = []
        for i in range(self.n_layers):
            if self.layer_side == Side.BUY:
                # Layer bids below mid, each level further from mid
                price = round(ref_price * (1 - self.layer_spacing * (i + 1)), 4)
            else:
                # Layer asks above mid, each level further from mid
                price = round(ref_price * (1 + self.layer_spacing * (i + 1)), 4)

            price = max(price, 0.01)
            order_id = str(uuid.uuid4())

            self._pending.append(LayerEvent(
                order_id=order_id,
                placed_tick=tick,
                cancel_at_tick=tick + self.cancellation_delay_ticks,
            ))

            orders.append(Order(
                order_id=order_id,
                trader_id=self.trader_id,
                side=self.layer_side,
                price=price,
                quantity=self.layer_size,
                tick=tick,
                order_type="limit",
            ))

        return orders

    def decide(self, book_state: BookState, tick: int) -> Order | None:
        """
        Standard Agent interface — returns the first layer order only.
        For full layering behaviour, the simulation loop should call decide_batch().
        """
        orders = self.decide_batch(book_state, tick)
        return orders[0] if orders else None

    def get_cancels(self, tick: int) -> list[Order]:
        """Return cancel instructions for all layers whose delay has elapsed."""
        due = [e for e in self._pending if e.cancel_at_tick <= tick]
        self._pending = [e for e in self._pending if e.cancel_at_tick > tick]

        return [
            Order(
                order_id=event.order_id,
                trader_id=self.trader_id,
                side=Side.BUY,  # irrelevant for cancel
                price=1.0,
                quantity=1,
                tick=tick,
                order_type="cancel",
            )
            for event in due
        ]

    def log_activity(self) -> dict:
        """Return a manipulation_events row for this agent's window."""
        return {
            "event_id": str(uuid.uuid4()),
            "persona": self.PERSONA,
            "start_tick": self.start_tick,
            "end_tick": self.end_tick,
            "trader_ids": [self.trader_id],
            "parameters": self.parameters_snapshot(),
        }

"""
src/core/agents/mean_reversion_trader.py
=========================================
Mean Reversion Trader — fundamental value anchor agent.

Represents commercial/hedging participants with a fundamental view on
commodity value. Corresponds to CFTC CoT "commercial" category (~30%).

Mechanism:
  Maintains a view of the asset's fundamental value (provided by the
  simulation's FundamentalValueProcess). When the market price deviates
  from the fundamental by more than a threshold, this agent fades the move:
  buys when market price is below fundamental, sells when above.

This is the core stabilising force of the ABM — mean reversion is an
EMERGENT property of agent incentives responding to the fundamental
value signal, NOT an imposed OU formula on the price series.
(ABM Project Plan Section 3.1 — "emergence not imposition" principle)
"""

from __future__ import annotations

import uuid

import numpy as np

from src.core.agents.base import Agent
from src.core.order_book import BookState, Order, Side


class MeanReversionTrader(Agent):
    """
    Fundamental-value-anchored agent that fades price deviations.

    The fundamental value is provided by the simulation loop (FundamentalValueProcess)
    and updated each tick via set_fundamental_value(). This agent does not compute
    the fundamental value itself — it receives it as a signal.

    Args:
        reversion_threshold: Minimum |price - fundamental| / fundamental to trigger action.
        order_size_base: Base order quantity.
        arrival_rate_lambda: Per-tick activation probability.
    """

    def __init__(
        self,
        trader_id: str,
        rng: np.random.Generator,
        *,
        reversion_threshold: float = 0.005,
        order_size_base: int = 40,
        arrival_rate_lambda: float = 0.25,
    ) -> None:
        super().__init__(trader_id, rng)
        self.reversion_threshold = reversion_threshold
        self.order_size_base = order_size_base
        self.arrival_rate_lambda = arrival_rate_lambda

        self._fundamental_value: float | None = None

    def set_fundamental_value(self, value: float) -> None:
        """
        Update the agent's view of the fundamental value.
        Called by the simulation loop each tick from FundamentalValueProcess.
        """
        if value > 0:
            self._fundamental_value = value

    def decide(self, book_state: BookState, tick: int) -> Order | None:
        # --- Activation ---
        if self.rng.random() > self.arrival_rate_lambda:
            return None

        if self._fundamental_value is None:
            return None  # no fundamental value signal yet

        # --- Reference price ---
        ref_price = book_state.mid_price or book_state.last_trade_price
        if ref_price is None or ref_price <= 0:
            return None

        # --- Compute deviation from fundamental ---
        deviation = (ref_price - self._fundamental_value) / self._fundamental_value

        if abs(deviation) < self.reversion_threshold:
            return None  # price is close enough to fundamental; no action

        # --- Fade the move ---
        if deviation > 0:
            # Price above fundamental → sell (fade the overvaluation)
            side = Side.SELL
            price = ref_price - self.rng.uniform(0, 0.3)
        else:
            # Price below fundamental → buy (fade the undervaluation)
            side = Side.BUY
            price = ref_price + self.rng.uniform(0, 0.3)

        price = round(max(price, 0.01), 4)

        # Order size scales with deviation strength
        deviation_strength = min(abs(deviation) / self.reversion_threshold, 5.0)
        quantity = int(self.order_size_base * deviation_strength)
        quantity = max(1, quantity)

        return Order(
            order_id=str(uuid.uuid4()),
            trader_id=self.trader_id,
            side=side,
            price=price,
            quantity=quantity,
            tick=tick,
            order_type="limit",
        )

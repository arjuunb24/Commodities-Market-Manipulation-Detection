"""
src/core/agents/occasional_trader.py
====================================
Occasional (discretionary) trader: a legitimate account that trades rarely and
in small size, at random times, leaning against its own position.

Why it exists: manipulator accounts trade lightly outside their manipulation
windows (cover, see manipulation/common.py). If no ordinary participant traded
that way, the detector could learn to recognise the ACCOUNTS by this background
fingerprint (sparse, small, irregular orders) instead of the manipulation, which
is what happened before: most "no matching episode" alerts were manipulator
accounts flagged while doing nothing manipulative. Real markets are full of such
occasional traders, so both share exactly the same order-generating code
(discretionary_order) and the fingerprint identifies nobody.
"""

from __future__ import annotations

import numpy as np

from src.core.agents.base import Agent
from src.core.order_book import BookState, Order, Side


def discretionary_order(agent: Agent, book_state: BookState, tick: int, rate: float) -> Order | None:
    """
    One small limit order near the mid with probability `rate` per tick, leaning
    against the account's net position so it stays roughly flat. Shared by
    OccasionalTrader and ManipulatorAccount (background flow).
    """
    if agent.rng.random() > rate:
        return None
    ref = book_state.mid_price or book_state.last_trade_price
    if ref is None or ref <= 0:
        return None
    p_buy = 0.5 - 0.3 * float(np.clip(agent.net_position / 200.0, -1, 1))
    side = Side.BUY if agent.rng.random() < p_buy else Side.SELL
    price = ref * (1 + agent.rng.uniform(-0.002, 0.002))
    qty = int(agent.rng.integers(5, 40))
    return agent.limit(side, price, qty, tick)


class OccasionalTrader(Agent):
    """Legitimate low-frequency discretionary trader."""

    mean_order_ttl = 40.0

    def __init__(self, trader_id: str, rng: np.random.Generator, *, activity_rate: float = 0.05) -> None:
        super().__init__(trader_id, rng)
        self.activity_rate = float(activity_rate)
        self.net_position = 0

    def update_inventory(self, side: Side, quantity: int) -> None:
        self.net_position += quantity if side == Side.BUY else -quantity

    def decide(self, book_state: BookState, tick: int) -> Order | None:
        return discretionary_order(self, book_state, tick, self.activity_rate)

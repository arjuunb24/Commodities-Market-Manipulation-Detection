"""
src/core/agents/market_maker.py
================================
Market Maker — continuous two-sided quote agent.

The market maker is effectively always active (high λ). It provides
two-sided quotes around the mid-price with an inventory-skew adjustment:
when the maker has accumulated net long inventory, it skews quotes to
sell more aggressively (and vice versa), which naturally mean-reverts
the maker's inventory over time.

Mechanism:
  Each tick: atomically cancel existing quotes, then place new bid + ask.
  This is the "requote" pattern — logged as a requote pair in the order_log.
  Requote atomicity is implemented at the book level: the simulation loop
  calls cancel_order() then submit() for both sides in the same tick batch.

Role in the simulation:
  - Primary provider of liquidity depth
  - Constrains the bid-ask spread (signals market quality)
  - Inventory accumulation under manipulation signals price impact

Realism notes:
  - Inventory skew is expressed in half-spreads (Avellaneda-Stoikov style), not
    as a percentage of price, so a modest inventory cannot move quotes by %s.
  - Quotes lean toward the visible order-book imbalance, as real liquidity
    providers do. This is precisely why spoofed depth is profitable: it moves
    the quotes other participants trade against.
"""

from __future__ import annotations


import numpy as np

from src.core.agents.base import Agent
from src.core.order_book import BookState, Order, Side


class MarketMaker(Agent):
    """
    Inventory-skewed, imbalance-aware two-sided market maker.

    Args:
        spread_ticks: Half-spread from the reference price, in price units.
        inventory_skew_factor: Skew (in half-spreads) at |inventory| == max_inventory.
        max_inventory: Absolute inventory cap; beyond it the maker quotes one side only.
        order_size: Quote size per side.
        arrival_rate_lambda: Per-tick requote probability.
        imbalance_lean: Shift (in half-spreads) toward a fully one-sided book.
    """

    def __init__(
        self,
        trader_id: str,
        rng: np.random.Generator,
        *,
        spread_ticks: float = 2.0,
        inventory_skew_factor: float = 1.0,
        max_inventory: int = 500,
        order_size: int = 40,
        arrival_rate_lambda: float = 0.95,
        imbalance_lean: float = 0.6,
    ) -> None:
        super().__init__(trader_id, rng)
        self.spread_ticks = spread_ticks
        self.inventory_skew_factor = inventory_skew_factor
        self.max_inventory = max_inventory
        self.order_size = order_size
        self.arrival_rate_lambda = arrival_rate_lambda
        self.imbalance_lean = imbalance_lean
        self.net_inventory: int = 0
        self._pending_bid_id: str | None = None
        self._pending_ask_id: str | None = None

    def update_inventory(self, trade_side: Side, quantity: int) -> None:
        """Update net inventory after a fill (called by the simulation loop)."""
        if trade_side == Side.BUY:
            self.net_inventory += quantity
        else:
            self.net_inventory -= quantity

    def decide(self, book_state: BookState, tick: int) -> Order | None:
        """Single-order interface (bid only). The simulation uses decide_pair()."""
        _, orders = self.decide_pair(book_state, tick)
        return next((o for o in orders if o.side == Side.BUY), None)

    def _quote_center(self, ref_price: float, book_state: BookState) -> float:
        inv_frac = float(np.clip(self.net_inventory / max(1, self.max_inventory), -1.0, 1.0))
        skew = self.inventory_skew_factor * inv_frac * self.spread_ticks
        lean = self.imbalance_lean * book_state.imbalance() * self.spread_ticks
        return ref_price - skew + lean

    def decide_pair(
        self, book_state: BookState, tick: int
    ) -> tuple[list[str], list[Order]]:
        """
        Requote: cancel the existing resting quotes, place a new bid + ask.

        Returns:
            cancel_ids: order_ids to cancel (previous bid/ask)
            new_orders: new bid/ask Orders (one side is dropped at the inventory cap)
        """
        if self.rng.random() > self.arrival_rate_lambda:
            return [], []

        ref_price = book_state.mid_price or book_state.last_trade_price
        if ref_price is None or ref_price <= 0:
            return [], []

        cancel_ids = [oid for oid in (self._pending_bid_id, self._pending_ask_id) if oid is not None]
        self._pending_bid_id = self._pending_ask_id = None

        center = self._quote_center(ref_price, book_state)
        bid_price = self.snap(max(center - self.spread_ticks, self.tick_size))
        ask_price = self.snap(max(center + self.spread_ticks, bid_price + self.tick_size))
        if ask_price <= bid_price:
            ask_price = self.snap(bid_price + self.tick_size)

        new_orders: list[Order] = []
        if self.net_inventory < self.max_inventory:
            bid = self.limit(Side.BUY, bid_price, self.order_size, tick)
            self._pending_bid_id = bid.order_id
            new_orders.append(bid)
        if self.net_inventory > -self.max_inventory:
            ask = self.limit(Side.SELL, ask_price, self.order_size, tick)
            self._pending_ask_id = ask.order_id
            new_orders.append(ask)
        return cancel_ids, new_orders

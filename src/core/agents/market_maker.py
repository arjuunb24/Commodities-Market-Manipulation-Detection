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
"""

from __future__ import annotations

import uuid

import numpy as np

from src.core.agents.base import Agent
from src.core.order_book import BookState, Order, Side


class MarketMaker(Agent):
    """
    Inventory-skewed two-sided market maker.

    Args:
        spread_ticks: Half-spread from mid-price in price units.
        inventory_skew_factor: Per-unit inventory skew (shifts quotes toward inventory reduction).
        max_inventory: Absolute inventory cap; above this, maker stops adding on one side.
        order_size: Quote size per side per tick.
        arrival_rate_lambda: Per-tick activation probability (default ~0.95 = near-always).
    """

    def __init__(
        self,
        trader_id: str,
        rng: np.random.Generator,
        *,
        spread_ticks: float = 2.0,
        inventory_skew_factor: float = 0.1,
        max_inventory: int = 500,
        order_size: int = 20,
        arrival_rate_lambda: float = 0.95,
    ) -> None:
        super().__init__(trader_id, rng)
        self.spread_ticks = spread_ticks
        self.inventory_skew_factor = inventory_skew_factor
        self.max_inventory = max_inventory
        self.order_size = order_size
        self.arrival_rate_lambda = arrival_rate_lambda

        self.net_inventory: int = 0               # positive = net long
        self._pending_bid_id: str | None = None   # track current resting bid for cancel
        self._pending_ask_id: str | None = None   # track current resting ask for cancel

    def update_inventory(self, trade_side: Side, quantity: int) -> None:
        """
        Update net inventory after a fill on this maker's resting order.
        Called by the simulation loop after processing each Trade involving this maker.
        """
        if trade_side == Side.BUY:
            self.net_inventory += quantity   # maker was the resting buyer
        else:
            self.net_inventory -= quantity   # maker was the resting seller

    def decide(self, book_state: BookState, tick: int) -> Order | None:
        """
        Return a new bid order. The simulation loop calls this twice
        (once for bid, once for ask) or calls the dedicated decide_pair() method.

        NOTE: For the market maker, the simulation loop must use decide_pair()
        to get both quotes atomically. This single decide() returns the bid.
        """
        if self.rng.random() > self.arrival_rate_lambda:
            return None

        ref_price = book_state.mid_price or book_state.last_trade_price
        if ref_price is None or ref_price <= 0:
            return None

        return self._make_bid(ref_price, tick)

    def decide_pair(
        self, book_state: BookState, tick: int
    ) -> tuple[list[str], list[Order]]:
        """
        Atomic requote: cancel existing resting quotes, place new bid + ask.

        Returns:
            cancel_ids: list of order_ids to cancel (at most 2: pending bid + ask)
            new_orders: list of new Orders to submit (at most 2: new bid + ask)

        The simulation loop processes cancels first, then new orders, in the
        same tick — ensuring atomicity within the book for this tick.
        """
        if self.rng.random() > self.arrival_rate_lambda:
            return [], []

        ref_price = book_state.mid_price or book_state.last_trade_price
        if ref_price is None or ref_price <= 0:
            return [], []

        # Collect cancel instructions for existing resting orders
        cancel_ids: list[str] = []
        if self._pending_bid_id is not None:
            cancel_ids.append(self._pending_bid_id)
        if self._pending_ask_id is not None:
            cancel_ids.append(self._pending_ask_id)

        # Inventory skew: shift the ref_price slightly to discourage inventory accumulation
        skew = self.inventory_skew_factor * self.net_inventory
        skewed_mid = ref_price - skew  # positive inventory → skew bids down, asks down

        # Generate new bid
        bid_id = str(uuid.uuid4())
        bid_price = round(max(skewed_mid - self.spread_ticks, 0.01), 4)

        # Generate new ask
        ask_id = str(uuid.uuid4())
        ask_price = round(skewed_mid + self.spread_ticks, 4)

        # Do not quote if inventory is too extreme
        new_orders: list[Order] = []

        if self.net_inventory < self.max_inventory:
            new_orders.append(Order(
                order_id=bid_id,
                trader_id=self.trader_id,
                side=Side.BUY,
                price=bid_price,
                quantity=self.order_size,
                tick=tick,
                order_type="limit",
            ))
            self._pending_bid_id = bid_id

        if self.net_inventory > -self.max_inventory:
            new_orders.append(Order(
                order_id=ask_id,
                trader_id=self.trader_id,
                side=Side.SELL,
                price=ask_price,
                quantity=self.order_size,
                tick=tick,
                order_type="limit",
            ))
            self._pending_ask_id = ask_id

        return cancel_ids, new_orders

    def _make_bid(self, ref_price: float, tick: int) -> Order | None:
        if self.net_inventory >= self.max_inventory:
            return None
        skew = self.inventory_skew_factor * self.net_inventory
        bid_price = round(max(ref_price - skew - self.spread_ticks, 0.01), 4)
        bid_id = str(uuid.uuid4())
        self._pending_bid_id = bid_id
        return Order(
            order_id=bid_id,
            trader_id=self.trader_id,
            side=Side.BUY,
            price=bid_price,
            quantity=self.order_size,
            tick=tick,
            order_type="limit",
        )

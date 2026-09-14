"""
src/core/order_book.py
======================
Continuous Double Auction Limit Order Book (LOB) matching engine.

DESIGN PRINCIPLES (from ABM Project Plan Section 3.1):
  - Pure data structures + control flow. Zero ML, zero randomness.
  - Price-time priority: bids sorted descending by price, then ascending by time;
    asks sorted ascending by price, then ascending by time.
  - Price ALWAYS emerges from two real orders crossing — never imposed externally.
  - This is the single source of truth. Everything else reads from its logs.

EDGE CASES (Implementation Plan Section 5.2 — all required):
  E1. Multi-level order walk: log one Trade per resting order consumed.
  E2. Incoming buy at exact best-bid price → rests (time priority), never self-matches.
  E3. Cancel of filled/unknown order_id → no-op, log rejected_cancel, never raise.
  E4. Zero/negative quantity → ValueError in test mode; log+drop in sim run.
  E5. Tick collision tie-break → deterministic insertion order in activation queue.
  E6. Market maker requote → atomic cancel-and-replace, logged as requote pair.
  E7. Self-trade prevention for all non-wash-trading agents (enforced at agent-logic level;
      the book itself is told about wash-trading pairs via allow_self_trade flag).
  E8. Zero-persona run → valid zero-row manipulation_events (not a missing file).
"""

from __future__ import annotations

import heapq
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Literal

import pandas as pd


# ===========================================================================
# Enumerations
# ===========================================================================


class Side(str, Enum):
    """Order side: buy or sell."""
    BUY = "buy"
    SELL = "sell"


class EventType(str, Enum):
    """All legal order lifecycle event types for the order_log."""
    PLACED = "placed"
    CANCELLED = "cancelled"
    FILLED = "filled"
    PARTIAL_FILL = "partial_fill"
    REJECTED_CANCEL = "rejected_cancel"  # cancel of already-filled / unknown order_id
    REQUOTE = "requote"                  # market maker atomic cancel-and-replace


# ===========================================================================
# Data classes
# ===========================================================================


@dataclass
class Order:
    """
    A single order submitted to the order book.

    Fields:
        order_id:    Unique identifier (typically a UUID string).
        trader_id:   Identifier of the submitting trader.
        side:        Buy or sell.
        price:       Limit price. Must be > 0.
        quantity:    Remaining quantity. Must be > 0.
        tick:        Simulation tick at which this order was submitted.
        order_type:  "limit" for a new resting order, "cancel" for a cancel instruction.
    """
    order_id: str
    trader_id: str
    side: Side
    price: float
    quantity: int
    tick: int
    order_type: Literal["limit", "cancel"]

    def __post_init__(self) -> None:
        if self.order_type == "limit":
            if self.price <= 0:
                raise ValueError(
                    f"Order {self.order_id}: price must be > 0, got {self.price}"
                )
            if self.quantity <= 0:
                raise ValueError(
                    f"Order {self.order_id}: quantity must be > 0, got {self.quantity}"
                )


@dataclass
class Trade:
    """
    A single executed trade — produced by OrderBook.submit().

    One Trade is generated per resting order consumed (not one per incoming order).
    This is required by Implementation Plan Section 5.2, Edge Case E1.
    """
    trade_id: str
    tick: int
    price: float           # execution price = resting order's price (price-time priority)
    quantity: int          # quantity matched in this trade
    buyer_trader_id: str
    seller_trader_id: str

    def to_dict(self) -> dict:
        return {
            "trade_id": self.trade_id,
            "tick": self.tick,
            "price": self.price,
            "quantity": self.quantity,
            "buyer_trader_id": self.buyer_trader_id,
            "seller_trader_id": self.seller_trader_id,
        }


@dataclass
class OrderEvent:
    """
    A single entry in the order_log — one per order lifecycle event.
    """
    order_id: str
    trader_id: str
    side: Side | None      # None for cancel/rejected_cancel events
    price: float | None    # None for cancel/rejected_cancel events
    quantity: int | None   # None for cancel/rejected_cancel events
    tick: int
    event_type: EventType

    @classmethod
    def from_order(cls, order: Order, event_type: EventType) -> "OrderEvent":
        return cls(
            order_id=order.order_id,
            trader_id=order.trader_id,
            side=order.side,
            price=order.price,
            quantity=order.quantity,
            tick=order.tick,
            event_type=event_type,
        )

    @classmethod
    def rejected_cancel(cls, order_id: str, trader_id: str, tick: int) -> "OrderEvent":
        """Factory for rejected_cancel events (filled/unknown order_id cancel)."""
        return cls(
            order_id=order_id,
            trader_id=trader_id,
            side=None,
            price=None,
            quantity=None,
            tick=tick,
            event_type=EventType.REJECTED_CANCEL,
        )

    def to_dict(self) -> dict:
        return {
            "order_id": self.order_id,
            "trader_id": self.trader_id,
            "side": self.side.value if self.side is not None else None,
            "price": self.price,
            "quantity": self.quantity,
            "tick": self.tick,
            "event_type": self.event_type.value,
        }


@dataclass
class BookState:
    """
    Snapshot of the order book's current state, passed to every agent's decide() call.

    This is a read-only view — agents must never modify the book directly.
    """
    best_bid: float | None           # highest resting bid price (None if book is empty)
    best_ask: float | None           # lowest resting ask price (None if book is empty)
    mid_price: float | None          # (best_bid + best_ask) / 2; None if either side empty
    bid_depth: list[tuple[float, int]]   # [(price, total_qty), ...] top-N bids, descending
    ask_depth: list[tuple[float, int]]   # [(price, total_qty), ...] top-N asks, ascending
    last_trade_price: float | None   # price of the most recent executed trade
    tick: int                        # current simulation tick


# ===========================================================================
# Price-time priority queue helpers
# ===========================================================================


@dataclass(order=True)
class _BidEntry:
    """
    Entry in the bid heap (max-heap implemented via negated price).
    Sorted by: highest price first, then lowest tick first (time priority).
    """
    neg_price: float      # negate to turn min-heap into max-heap
    tick: int             # insertion tick (lower = higher time priority)
    order_id: str = field(compare=False)
    trader_id: str = field(compare=False)
    quantity: int = field(compare=False)
    price: float = field(compare=False)


@dataclass(order=True)
class _AskEntry:
    """
    Entry in the ask heap (min-heap — naturally ascending price order).
    Sorted by: lowest price first, then lowest tick first (time priority).
    """
    price: float
    tick: int
    order_id: str = field(compare=False)
    trader_id: str = field(compare=False)
    quantity: int = field(compare=False)


# ===========================================================================
# OrderBook
# ===========================================================================


class OrderBook:
    """
    Continuous Double Auction Limit Order Book with price-time priority matching.

    State:
      _bids: min-heap of _BidEntry (negated price → effectively max-heap)
      _asks: min-heap of _AskEntry (natural min-heap on price)
      _resting: dict[order_id → {price, quantity, side, trader_id, tick}]
                Fast lookup for cancels and for the book_state() depth snapshot.
      _cancelled: set of order_ids that have been fully cancelled or filled
                  Used to implement lazy deletion from the heaps.

    Lazy deletion:
      Heaps don't support efficient removal, so cancelled/filled orders remain
      in the heap until they reach the top and are found to be in _cancelled —
      at that point they are discarded (standard "lazy deletion" pattern).
      This is O(log n) amortised for all operations and is correct.
    """

    def __init__(self, initial_price: float | None = None) -> None:
        self._bids: list[_BidEntry] = []   # min-heap (negated price)
        self._asks: list[_AskEntry] = []   # min-heap

        # order_id → full order details (for cancel lookup + depth computation)
        self._resting: dict[str, dict] = {}

        # set of order_ids that are no longer active (filled or cancelled)
        self._inactive: set[str] = set()

        self._last_trade_price: float | None = initial_price

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def submit(
        self,
        order: Order,
        *,
        simulation_mode: bool = True,
    ) -> tuple[list[Trade], list[OrderEvent]]:
        """
        Process one order or cancel instruction.

        Args:
            order: The Order to process.
            simulation_mode: If True, zero/negative quantity errors are logged
                             and the order is dropped (never crash the whole run).
                             If False (test mode), they raise ValueError immediately.

        Returns:
            A tuple of:
                trades: list[Trade] — executed trades (empty if order rested or was cancel).
                events: list[OrderEvent] — order log events generated by this submit call.

        Note: This method does NOT write to any file. The simulation loop is
        responsible for collecting trades + events and writing them to disk.
        """
        # --- Validate quantity / price (Edge Case E4) ---
        if order.order_type == "limit":
            if order.quantity <= 0 or order.price <= 0:
                if simulation_mode:
                    return [], [OrderEvent(
                        order_id=order.order_id,
                        trader_id=order.trader_id,
                        side=order.side if order.side else None,
                        price=order.price if order.price > 0 else None,
                        quantity=None,
                        tick=order.tick,
                        event_type=EventType.REJECTED_CANCEL,  # reuse for rejected invalid
                    )]
                raise ValueError(
                    f"Order {order.order_id}: price and quantity must be > 0 "
                    f"(got price={order.price}, quantity={order.quantity})"
                )

        if order.order_type == "cancel":
            return self._process_cancel(order)
        else:
            return self._process_limit(order)

    def best_bid(self) -> float | None:
        """Return the highest resting bid price, or None if the bid side is empty."""
        self._clean_top(self._bids, is_bid=True)
        if not self._bids:
            return None
        return self._bids[0].price  # _BidEntry.price (not neg_price)

    def best_ask(self) -> float | None:
        """Return the lowest resting ask price, or None if the ask side is empty."""
        self._clean_top(self._asks, is_bid=False)
        if not self._asks:
            return None
        return self._asks[0].price

    def book_state(self, depth: int = 5) -> BookState:
        """
        Snapshot of the current book state for agent decision-making.
        Called by every agent's decide() call each tick.

        Args:
            depth: Number of price levels to include in bid_depth / ask_depth.

        Returns:
            BookState — read-only snapshot.
        """
        bb = self.best_bid()
        ba = self.best_ask()
        mid = (bb + ba) / 2.0 if (bb is not None and ba is not None) else None

        bid_levels = self._aggregate_depth(self._bids, is_bid=True, depth=depth)
        ask_levels = self._aggregate_depth(self._asks, is_bid=False, depth=depth)

        # tick is injected by the simulation loop; we don't track it in the book
        return BookState(
            best_bid=bb,
            best_ask=ba,
            mid_price=mid,
            bid_depth=bid_levels,
            ask_depth=ask_levels,
            last_trade_price=self._last_trade_price,
            tick=-1,  # simulation loop overwrites this
        )

    def cancel_order(self, order_id: str, trader_id: str, tick: int) -> list[OrderEvent]:
        """
        Convenience method to cancel a specific order_id directly.
        Returns list containing either a CANCELLED or REJECTED_CANCEL event.
        """
        cancel_order = Order(
            order_id=order_id,
            trader_id=trader_id,
            side=Side.BUY,      # side is irrelevant for cancels
            price=1.0,          # price is irrelevant for cancels
            quantity=1,         # quantity is irrelevant for cancels
            tick=tick,
            order_type="cancel",
        )
        _, events = self.submit(cancel_order)
        return events

    # -----------------------------------------------------------------------
    # Private: matching logic
    # -----------------------------------------------------------------------

    def _process_limit(self, order: Order) -> tuple[list[Trade], list[OrderEvent]]:
        """
        Process a new limit order: try to match, then rest if not fully filled.
        """
        trades: list[Trade] = []
        events: list[OrderEvent] = []

        # Log the placed event
        events.append(OrderEvent.from_order(order, EventType.PLACED))

        remaining_qty = order.quantity

        if order.side == Side.BUY:
            # Walk the ask side while we can match (price crosses)
            while remaining_qty > 0 and self._asks:
                self._clean_top(self._asks, is_bid=False)
                if not self._asks:
                    break

                best_ask_entry = self._asks[0]

                # Self-trade prevention (Edge Case E7):
                # The order book does NOT enforce this — agents must avoid
                # placing orders that cross their own resting orders.
                # For the wash-trading persona, crossing own orders is intentional.

                if order.price < best_ask_entry.price:
                    break  # order does not cross best ask → rest

                # Match at the resting (ask) price — price-time priority
                match_qty = min(remaining_qty, best_ask_entry.quantity)
                trade = Trade(
                    trade_id=str(uuid.uuid4()),
                    tick=order.tick,
                    price=best_ask_entry.price,   # execution at resting order's price
                    quantity=match_qty,
                    buyer_trader_id=order.trader_id,
                    seller_trader_id=best_ask_entry.trader_id,
                )
                trades.append(trade)
                self._last_trade_price = trade.price

                remaining_qty -= match_qty
                new_resting_qty = best_ask_entry.quantity - match_qty

                if new_resting_qty == 0:
                    # Resting order fully filled → remove from heap + resting dict
                    heapq.heappop(self._asks)
                    self._inactive.add(best_ask_entry.order_id)
                    self._resting.pop(best_ask_entry.order_id, None)
                    events.append(OrderEvent(
                        order_id=best_ask_entry.order_id,
                        trader_id=best_ask_entry.trader_id,
                        side=Side.SELL,
                        price=best_ask_entry.price,
                        quantity=match_qty,
                        tick=order.tick,
                        event_type=EventType.FILLED,
                    ))
                else:
                    # Resting order partially filled → update quantity
                    self._asks[0] = _AskEntry(
                        price=best_ask_entry.price,
                        tick=best_ask_entry.tick,
                        order_id=best_ask_entry.order_id,
                        trader_id=best_ask_entry.trader_id,
                        quantity=new_resting_qty,
                    )
                    heapq.heapify(self._asks)
                    self._resting[best_ask_entry.order_id]["quantity"] = new_resting_qty
                    events.append(OrderEvent(
                        order_id=best_ask_entry.order_id,
                        trader_id=best_ask_entry.trader_id,
                        side=Side.SELL,
                        price=best_ask_entry.price,
                        quantity=match_qty,
                        tick=order.tick,
                        event_type=EventType.PARTIAL_FILL,
                    ))

        else:  # SELL order
            while remaining_qty > 0 and self._bids:
                self._clean_top(self._bids, is_bid=True)
                if not self._bids:
                    break

                best_bid_entry = self._bids[0]

                if order.price > best_bid_entry.price:
                    break  # order does not cross best bid → rest

                match_qty = min(remaining_qty, best_bid_entry.quantity)
                trade = Trade(
                    trade_id=str(uuid.uuid4()),
                    tick=order.tick,
                    price=best_bid_entry.price,   # execution at resting bid price
                    quantity=match_qty,
                    buyer_trader_id=best_bid_entry.trader_id,
                    seller_trader_id=order.trader_id,
                )
                trades.append(trade)
                self._last_trade_price = trade.price

                remaining_qty -= match_qty
                new_resting_qty = best_bid_entry.quantity - match_qty

                if new_resting_qty == 0:
                    heapq.heappop(self._bids)
                    self._inactive.add(best_bid_entry.order_id)
                    self._resting.pop(best_bid_entry.order_id, None)
                    events.append(OrderEvent(
                        order_id=best_bid_entry.order_id,
                        trader_id=best_bid_entry.trader_id,
                        side=Side.BUY,
                        price=best_bid_entry.price,
                        quantity=match_qty,
                        tick=order.tick,
                        event_type=EventType.FILLED,
                    ))
                else:
                    self._bids[0] = _BidEntry(
                        neg_price=-best_bid_entry.price,
                        tick=best_bid_entry.tick,
                        order_id=best_bid_entry.order_id,
                        trader_id=best_bid_entry.trader_id,
                        quantity=new_resting_qty,
                        price=best_bid_entry.price,
                    )
                    heapq.heapify(self._bids)
                    self._resting[best_bid_entry.order_id]["quantity"] = new_resting_qty
                    events.append(OrderEvent(
                        order_id=best_bid_entry.order_id,
                        trader_id=best_bid_entry.trader_id,
                        side=Side.BUY,
                        price=best_bid_entry.price,
                        quantity=match_qty,
                        tick=order.tick,
                        event_type=EventType.PARTIAL_FILL,
                    ))

        # Log the incoming order's own fill/rest event
        if remaining_qty == 0:
            events.append(OrderEvent.from_order(order, EventType.FILLED))
        elif remaining_qty < order.quantity:
            # Partially filled → log partial fill then rest the remainder
            partial_order = Order(
                order_id=order.order_id,
                trader_id=order.trader_id,
                side=order.side,
                price=order.price,
                quantity=order.quantity - remaining_qty,
                tick=order.tick,
                order_type="limit",
            )
            events.append(OrderEvent.from_order(partial_order, EventType.PARTIAL_FILL))
            self._rest_order(order, remaining_qty)
        else:
            # Not filled at all → rest on the book
            self._rest_order(order, remaining_qty)

        return trades, events

    def _rest_order(self, order: Order, quantity: int) -> None:
        """Place an order (or remaining quantity) onto the book."""
        if order.side == Side.BUY:
            entry = _BidEntry(
                neg_price=-order.price,
                tick=order.tick,
                order_id=order.order_id,
                trader_id=order.trader_id,
                quantity=quantity,
                price=order.price,
            )
            heapq.heappush(self._bids, entry)
        else:
            entry = _AskEntry(
                price=order.price,
                tick=order.tick,
                order_id=order.order_id,
                trader_id=order.trader_id,
                quantity=quantity,
            )
            heapq.heappush(self._asks, entry)

        self._resting[order.order_id] = {
            "price": order.price,
            "quantity": quantity,
            "side": order.side,
            "trader_id": order.trader_id,
            "tick": order.tick,
        }

    def _process_cancel(self, order: Order) -> tuple[list[Trade], list[OrderEvent]]:
        """
        Process a cancel instruction.

        Edge Case E3: cancel of a filled or unknown order_id → no-op,
        log rejected_cancel event, NEVER raise an exception.
        This is critical for the spoofing agent, which may attempt to cancel
        an order that was partially or fully filled before the cancel arrived.
        """
        target_id = order.order_id

        if target_id in self._inactive or target_id not in self._resting:
            # Filled, already cancelled, or was never in the book
            event = OrderEvent.rejected_cancel(
                order_id=target_id,
                trader_id=order.trader_id,
                tick=order.tick,
            )
            return [], [event]

        # Order is resting — mark inactive (lazy deletion from heap)
        resting_info = self._resting.pop(target_id)
        self._inactive.add(target_id)

        event = OrderEvent(
            order_id=target_id,
            trader_id=resting_info["trader_id"],
            side=resting_info["side"],
            price=resting_info["price"],
            quantity=resting_info["quantity"],
            tick=order.tick,
            event_type=EventType.CANCELLED,
        )
        return [], [event]

    # -----------------------------------------------------------------------
    # Private: heap maintenance
    # -----------------------------------------------------------------------

    def _clean_top(self, heap: list, *, is_bid: bool) -> None:
        """
        Remove inactive entries from the top of a heap (lazy deletion).
        Modifies heap in place.
        """
        while heap:
            top_id = heap[0].order_id
            if top_id in self._inactive:
                heapq.heappop(heap)
            else:
                break

    def _aggregate_depth(
        self,
        heap: list,
        *,
        is_bid: bool,
        depth: int,
    ) -> list[tuple[float, int]]:
        """
        Compute the depth-N price level aggregation for book_state().

        Returns a list of (price, total_quantity) tuples, bids descending,
        asks ascending, up to `depth` levels.
        Does NOT modify the heap.
        """
        levels: dict[float, int] = {}
        seen_inactive = set()

        for entry in heap:
            if entry.order_id in self._inactive:
                continue
            price = entry.price if is_bid else entry.price
            levels[price] = levels.get(price, 0) + entry.quantity
            if len(levels) >= depth * 3:  # approximate limit for performance
                break

        sorted_levels = sorted(levels.items(), key=lambda x: -x[0] if is_bid else x[0])
        return sorted_levels[:depth]

    # -----------------------------------------------------------------------
    # Snapshot utilities
    # -----------------------------------------------------------------------

    def to_dataframes(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Return the current resting book as (bids_df, asks_df) DataFrames.
        Useful for debugging and sanity-check plots.
        """
        bids = []
        asks = []
        for oid, info in self._resting.items():
            row = {"order_id": oid, **info, "side": info["side"].value}
            if info["side"] == Side.BUY:
                bids.append(row)
            else:
                asks.append(row)

        bids_df = pd.DataFrame(bids).sort_values("price", ascending=False) if bids else pd.DataFrame()
        asks_df = pd.DataFrame(asks).sort_values("price", ascending=True) if asks else pd.DataFrame()
        return bids_df, asks_df


# ===========================================================================
# Helpers
# ===========================================================================


def make_order_id() -> str:
    """Generate a unique order identifier."""
    return str(uuid.uuid4())


def make_event_id() -> str:
    """Generate a unique manipulation event identifier."""
    return str(uuid.uuid4())

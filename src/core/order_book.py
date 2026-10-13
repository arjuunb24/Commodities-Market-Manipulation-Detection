"""
src/core/order_book.py
======================
Continuous Double Auction Limit Order Book (LOB) matching engine.

DESIGN PRINCIPLES (from ABM Project Plan Section 3.1):
  - Pure data structures + control flow. Zero ML, zero randomness.
  - Price-time priority: bids sorted descending by price, then ascending by arrival;
    asks sorted ascending by price, then ascending by arrival.
  - Price ALWAYS emerges from two real orders crossing — never imposed externally.
  - This is the single source of truth. Everything else reads from its logs.

EDGE CASES (Implementation Plan Section 5.2 — all required):
  E1. Multi-level order walk: log one Trade per resting order consumed.
  E2. Incoming buy at exact best-bid price → rests (time priority), never self-matches.
  E3. Cancel of filled/unknown order_id → no-op, log rejected_cancel, never raise.
  E4. Zero/negative quantity → ValueError in test mode; log+drop in sim run.
  E5. Tick collision tie-break → strict arrival sequence number (deterministic).
  E6. Market maker requote → cancel-and-replace inside the same tick.
  E7. Self-trade prevention for all non-wash-trading agents (enforced at agent-logic level).
  E8. Zero-persona run → valid zero-row manipulation_events (not a missing file).

DETERMINISM (SRS FR-0.4):
  Trade ids are a per-book counter prefixed with `id_prefix`, so two runs with the
  same seed produce byte-identical trade logs (no uuid4 anywhere in the sim path).
"""

from __future__ import annotations

import heapq
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


def round_to_tick(price: float, tick_size: float) -> float:
    """Snap a price onto the instrument's tick grid (never below one tick)."""
    if tick_size <= 0:
        return round(price, 6)
    snapped = round(round(price / tick_size) * tick_size, 8)
    return max(snapped, tick_size)


# ===========================================================================
# Data classes
# ===========================================================================


@dataclass
class Order:
    """
    A single order submitted to the order book.

    Fields:
        order_id:    Unique identifier (deterministic: <prefix><trader_id>-<n>).
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
    `aggressor_side` is the side of the incoming (liquidity-taking) order — real
    exchange trade feeds publish this flag and surveillance relies on it.
    """
    trade_id: str
    tick: int
    price: float           # execution price = resting order's price (price-time priority)
    quantity: int          # quantity matched in this trade
    buyer_trader_id: str
    seller_trader_id: str
    aggressor_side: Side = Side.BUY

    def to_dict(self) -> dict:
        return {
            "trade_id": self.trade_id,
            "tick": self.tick,
            "price": self.price,
            "quantity": self.quantity,
            "buyer_trader_id": self.buyer_trader_id,
            "seller_trader_id": self.seller_trader_id,
            "aggressor_side": self.aggressor_side.value,
        }


@dataclass
class OrderEvent:
    """
    A single entry in the order_log — one per order lifecycle event.
    """
    order_id: str
    trader_id: str
    side: Side | None      # None for rejected_cancel events
    price: float | None    # None for rejected_cancel events
    quantity: int | None   # None for rejected_cancel events
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

    bid_band_volume / ask_band_volume: total resting volume within ±band of mid.
    This is the visible-liquidity signal that real order-book-imbalance (OBI)
    strategies react to, and the channel through which spoofing moves prices.
    """
    best_bid: float | None           # highest resting bid price (None if book is empty)
    best_ask: float | None           # lowest resting ask price (None if book is empty)
    mid_price: float | None          # (best_bid + best_ask) / 2; None if either side empty
    bid_depth: list[tuple[float, int]]   # [(price, total_qty), ...] top-N bids, descending
    ask_depth: list[tuple[float, int]]   # [(price, total_qty), ...] top-N asks, ascending
    last_trade_price: float | None   # price of the most recent executed trade
    tick: int                        # current simulation tick
    bid_band_volume: int = 0
    ask_band_volume: int = 0

    def imbalance(self) -> float:
        """Visible order-book imbalance in [-1, 1]; positive = bid-heavy."""
        bv, av = self.bid_band_volume, self.ask_band_volume
        if bv + av == 0:
            bv = sum(q for _, q in self.bid_depth)
            av = sum(q for _, q in self.ask_depth)
        total = bv + av
        return (bv - av) / total if total > 0 else 0.0


# ===========================================================================
# Price-time priority queue helpers
# ===========================================================================


@dataclass(order=True)
class _BidEntry:
    """Bid heap entry: highest price first, then earliest arrival (seq)."""
    neg_price: float
    seq: int
    order_id: str = field(compare=False)
    trader_id: str = field(compare=False)
    quantity: int = field(compare=False)
    price: float = field(compare=False)


@dataclass(order=True)
class _AskEntry:
    """Ask heap entry: lowest price first, then earliest arrival (seq)."""
    price: float
    seq: int
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
      _bids / _asks: heaps with lazy deletion of cancelled orders.
      _resting:      order_id → resting order details (cancel lookup).
      _bid_levels / _ask_levels: price → aggregate resting quantity, kept in sync
                     with every rest / fill / cancel so depth snapshots are exact.
    """

    def __init__(self, initial_price: float | None = None, id_prefix: str = "") -> None:
        self._bids: list[_BidEntry] = []
        self._asks: list[_AskEntry] = []
        self._resting: dict[str, dict] = {}
        self._inactive: set[str] = set()
        self._bid_levels: dict[float, int] = {}
        self._ask_levels: dict[float, int] = {}
        self._last_trade_price: float | None = initial_price
        self._id_prefix = id_prefix
        self._trade_seq = 0
        self._arrival_seq = 0

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
            (trades, events) generated by this submit call. The book never writes files.
        """
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
        return self._process_limit(order)

    def best_bid(self) -> float | None:
        """Return the highest resting bid price, or None if the bid side is empty."""
        self._clean_top(self._bids)
        return self._bids[0].price if self._bids else None

    def best_ask(self) -> float | None:
        """Return the lowest resting ask price, or None if the ask side is empty."""
        self._clean_top(self._asks)
        return self._asks[0].price if self._asks else None

    def is_resting(self, order_id: str) -> bool:
        """True while order_id still has quantity resting on the book."""
        return order_id in self._resting

    def book_state(self, depth: int = 5, band: float = 0.002) -> BookState:
        """
        Snapshot of the current book state for agent decision-making.

        Args:
            depth: Number of price levels to include in bid_depth / ask_depth.
            band:  Half-width (fraction of mid) for the band-volume imbalance signal.
        """
        bb = self.best_bid()
        ba = self.best_ask()
        mid = (bb + ba) / 2.0 if (bb is not None and ba is not None) else None

        bid_levels = heapq.nlargest(depth, self._bid_levels.items())
        ask_levels = heapq.nsmallest(depth, self._ask_levels.items())

        bid_band = ask_band = 0
        ref = mid if mid is not None else self._last_trade_price
        if ref is not None:
            lo, hi = ref * (1 - band), ref * (1 + band)
            bid_band = sum(q for p, q in self._bid_levels.items() if p >= lo)
            ask_band = sum(q for p, q in self._ask_levels.items() if p <= hi)

        return BookState(
            best_bid=bb,
            best_ask=ba,
            mid_price=mid,
            bid_depth=bid_levels,
            ask_depth=ask_levels,
            last_trade_price=self._last_trade_price,
            tick=-1,  # simulation loop overwrites this
            bid_band_volume=bid_band,
            ask_band_volume=ask_band,
        )

    def cancel_order(self, order_id: str, trader_id: str, tick: int) -> list[OrderEvent]:
        """Cancel a specific order_id; returns a CANCELLED or REJECTED_CANCEL event."""
        cancel = Order(
            order_id=order_id,
            trader_id=trader_id,
            side=Side.BUY,      # irrelevant for cancels
            price=1.0,
            quantity=1,
            tick=tick,
            order_type="cancel",
        )
        _, events = self.submit(cancel)
        return events

    # -----------------------------------------------------------------------
    # Private: matching logic
    # -----------------------------------------------------------------------

    def _next_trade_id(self) -> str:
        self._trade_seq += 1
        return f"{self._id_prefix}T{self._trade_seq}"

    def _process_limit(self, order: Order) -> tuple[list[Trade], list[OrderEvent]]:
        """Process a new limit order: match against the opposite side, rest any remainder."""
        trades: list[Trade] = []
        events: list[OrderEvent] = [OrderEvent.from_order(order, EventType.PLACED)]
        remaining_qty = order.quantity
        is_buy = order.side == Side.BUY
        heap = self._asks if is_buy else self._bids
        levels = self._ask_levels if is_buy else self._bid_levels
        resting_side = Side.SELL if is_buy else Side.BUY

        while remaining_qty > 0:
            self._clean_top(heap)
            if not heap:
                break
            top = heap[0]
            # Price-cross check (E2: equal price on the SAME side never matches —
            # we only ever look at the opposite heap here).
            if is_buy and order.price < top.price:
                break
            if not is_buy and order.price > top.price:
                break

            match_qty = min(remaining_qty, top.quantity)
            trade = Trade(
                trade_id=self._next_trade_id(),
                tick=order.tick,
                price=top.price,  # execution at the resting order's price
                quantity=match_qty,
                buyer_trader_id=order.trader_id if is_buy else top.trader_id,
                seller_trader_id=top.trader_id if is_buy else order.trader_id,
                aggressor_side=order.side,
            )
            trades.append(trade)
            self._last_trade_price = trade.price
            remaining_qty -= match_qty
            self._reduce_level(levels, top.price, match_qty)

            new_resting_qty = top.quantity - match_qty
            if new_resting_qty == 0:
                heapq.heappop(heap)
                self._inactive.add(top.order_id)
                self._resting.pop(top.order_id, None)
                event_type = EventType.FILLED
            else:
                # Ordering keys (price, seq) are unchanged, so the heap stays valid.
                top.quantity = new_resting_qty
                self._resting[top.order_id]["quantity"] = new_resting_qty
                event_type = EventType.PARTIAL_FILL
            events.append(OrderEvent(
                order_id=top.order_id,
                trader_id=top.trader_id,
                side=resting_side,
                price=top.price,
                quantity=match_qty,
                tick=order.tick,
                event_type=event_type,
            ))

        if remaining_qty == 0:
            events.append(OrderEvent.from_order(order, EventType.FILLED))
        else:
            if remaining_qty < order.quantity:
                filled = Order(
                    order_id=order.order_id,
                    trader_id=order.trader_id,
                    side=order.side,
                    price=order.price,
                    quantity=order.quantity - remaining_qty,
                    tick=order.tick,
                    order_type="limit",
                )
                events.append(OrderEvent.from_order(filled, EventType.PARTIAL_FILL))
            self._rest_order(order, remaining_qty)

        return trades, events

    def _rest_order(self, order: Order, quantity: int) -> None:
        """Place an order (or its remaining quantity) onto the book."""
        self._arrival_seq += 1
        if order.side == Side.BUY:
            heapq.heappush(self._bids, _BidEntry(
                neg_price=-order.price,
                seq=self._arrival_seq,
                order_id=order.order_id,
                trader_id=order.trader_id,
                quantity=quantity,
                price=order.price,
            ))
            self._bid_levels[order.price] = self._bid_levels.get(order.price, 0) + quantity
        else:
            heapq.heappush(self._asks, _AskEntry(
                price=order.price,
                seq=self._arrival_seq,
                order_id=order.order_id,
                trader_id=order.trader_id,
                quantity=quantity,
            ))
            self._ask_levels[order.price] = self._ask_levels.get(order.price, 0) + quantity
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
        """
        target_id = order.order_id
        if target_id in self._inactive or target_id not in self._resting:
            return [], [OrderEvent.rejected_cancel(target_id, order.trader_id, order.tick)]

        info = self._resting.pop(target_id)
        self._inactive.add(target_id)
        levels = self._bid_levels if info["side"] == Side.BUY else self._ask_levels
        self._reduce_level(levels, info["price"], info["quantity"])
        return [], [OrderEvent(
            order_id=target_id,
            trader_id=info["trader_id"],
            side=info["side"],
            price=info["price"],
            quantity=info["quantity"],
            tick=order.tick,
            event_type=EventType.CANCELLED,
        )]

    # -----------------------------------------------------------------------
    # Private: heap / level maintenance
    # -----------------------------------------------------------------------

    @staticmethod
    def _reduce_level(levels: dict[float, int], price: float, qty: int) -> None:
        left = levels.get(price, 0) - qty
        if left > 0:
            levels[price] = left
        else:
            levels.pop(price, None)

    def _clean_top(self, heap: list) -> None:
        """Pop inactive entries from the top of a heap (lazy deletion)."""
        while heap and heap[0].order_id in self._inactive:
            heapq.heappop(heap)

    # -----------------------------------------------------------------------
    # Snapshot utilities
    # -----------------------------------------------------------------------

    def to_dataframes(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Return the current resting book as (bids_df, asks_df) DataFrames."""
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

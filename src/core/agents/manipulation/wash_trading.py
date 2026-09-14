"""
src/core/agents/manipulation/wash_trading.py
=============================================
Wash Trading Pair — two colluding trader_ids crossing pre-arranged trades
with no net inventory change.

CFTC Definition:
  "Entering into, or purporting to enter into, transactions to give the
  appearance of purchases and sales but resulting in no net change in
  the trader's market position."

Grounded in:
  - CFTC v. Kraft Foods Group (2015): $16M wash trading settlement
  (see docs/enforcement_case_notes.md)

Mechanism:
  Two trader_ids (wash_buyer and wash_seller) simultaneously:
  1. Wash buyer places a buy limit order near mid-price.
  2. Wash seller places a matching sell limit order at same price.
  These cross in the order book, creating artificial volume with no net
  inventory change across the pair.

  In the order book, buyer_trader_id != seller_trader_id (two distinct
  trader_ids), so the trade appears legitimate at the book level.
  The manipulation is revealed by the net-position/gross-volume ratio
  feature: each account's gross volume is high, net position ≈ 0.

Note on self-cross:
  The book does NOT prevent this. The WashTradingPair uses two SEPARATE
  trader_ids for the two sides — this is intentional design.
"""

from __future__ import annotations

import uuid

import numpy as np

from src.core.agents.base import Agent
from src.core.order_book import BookState, Order, Side


class WashBuyer(Agent):
    """The buy-side of a wash trading pair. Must be paired with a WashSeller."""

    def __init__(
        self,
        trader_id: str,
        rng: np.random.Generator,
        partner_trader_id: str,
        *,
        trade_frequency: float = 0.5,
        price_deviation_from_mid: float = 0.001,
        start_tick: int = 0,
        end_tick: int = 10000,
    ) -> None:
        super().__init__(trader_id, rng)
        self.partner_trader_id = partner_trader_id
        self.trade_frequency = trade_frequency
        self.price_deviation_from_mid = price_deviation_from_mid
        self.start_tick = start_tick
        self.end_tick = end_tick

    def decide(self, book_state: BookState, tick: int) -> Order | None:
        if tick < self.start_tick or tick > self.end_tick:
            return None

        if self.rng.random() > self.trade_frequency:
            return None

        ref_price = book_state.mid_price or book_state.last_trade_price
        if ref_price is None or ref_price <= 0:
            return None

        # Place a buy slightly above mid (aggressive enough to cross the wash seller)
        buy_price = round(ref_price * (1 + self.price_deviation_from_mid), 4)

        return Order(
            order_id=str(uuid.uuid4()),
            trader_id=self.trader_id,
            side=Side.BUY,
            price=buy_price,
            quantity=int(self.rng.integers(10, 50)),
            tick=tick,
            order_type="limit",
        )


class WashSeller(Agent):
    """The sell-side of a wash trading pair. Must be paired with a WashBuyer."""

    def __init__(
        self,
        trader_id: str,
        rng: np.random.Generator,
        partner_trader_id: str,
        *,
        trade_frequency: float = 0.5,
        price_deviation_from_mid: float = 0.001,
        start_tick: int = 0,
        end_tick: int = 10000,
    ) -> None:
        super().__init__(trader_id, rng)
        self.partner_trader_id = partner_trader_id
        self.trade_frequency = trade_frequency
        self.price_deviation_from_mid = price_deviation_from_mid
        self.start_tick = start_tick
        self.end_tick = end_tick

    def decide(self, book_state: BookState, tick: int) -> Order | None:
        if tick < self.start_tick or tick > self.end_tick:
            return None

        if self.rng.random() > self.trade_frequency:
            return None

        ref_price = book_state.mid_price or book_state.last_trade_price
        if ref_price is None or ref_price <= 0:
            return None

        # Place a sell slightly below mid (aggressive enough to cross the wash buyer)
        sell_price = round(ref_price * (1 - self.price_deviation_from_mid), 4)

        return Order(
            order_id=str(uuid.uuid4()),
            trader_id=self.trader_id,
            side=Side.SELL,
            price=sell_price,
            quantity=int(self.rng.integers(10, 50)),
            tick=tick,
            order_type="limit",
        )


class WashTradingPair:
    """
    Factory that creates and manages a coordinated WashBuyer + WashSeller pair.

    Usage:
        pair = WashTradingPair.create(rng, pair_index=0, **params)
        agents = pair.agents  # [WashBuyer, WashSeller]
        event = pair.log_activity(start_tick)  # manipulation_events row

    The simulation loop adds both agents to its agent list.
    """

    PERSONA = "wash_trading"

    def __init__(
        self,
        buyer: WashBuyer,
        seller: WashSeller,
        start_tick: int,
        end_tick: int,
        parameters: dict,
    ) -> None:
        self.buyer = buyer
        self.seller = seller
        self.start_tick = start_tick
        self.end_tick = end_tick
        self.parameters = parameters
        self.agents: list[Agent] = [buyer, seller]

    @classmethod
    def create(
        cls,
        rng: np.random.Generator,
        pair_index: int,
        *,
        trade_frequency: float = 0.5,
        price_deviation_from_mid: float = 0.001,
        start_tick: int = 0,
        end_tick: int = 10000,
    ) -> "WashTradingPair":
        buyer_id = f"wash_buyer_{pair_index}"
        seller_id = f"wash_seller_{pair_index}"

        # Each agent gets its own sub-generator (rng.spawn(1)[0]) for independence
        buyer_rng, seller_rng = rng.spawn(2)

        buyer = WashBuyer(
            buyer_id, buyer_rng, seller_id,
            trade_frequency=trade_frequency,
            price_deviation_from_mid=price_deviation_from_mid,
            start_tick=start_tick,
            end_tick=end_tick,
        )
        seller = WashSeller(
            seller_id, seller_rng, buyer_id,
            trade_frequency=trade_frequency,
            price_deviation_from_mid=price_deviation_from_mid,
            start_tick=start_tick,
            end_tick=end_tick,
        )

        params = {
            "trade_frequency": trade_frequency,
            "price_deviation_from_mid": price_deviation_from_mid,
            "n_colluding_pairs": 1,  # this pair
        }
        return cls(buyer, seller, start_tick, end_tick, params)

    def log_activity(self) -> dict:
        """Return a manipulation_events row for this pair."""
        return {
            "event_id": str(uuid.uuid4()),
            "persona": self.PERSONA,
            "start_tick": self.start_tick,
            "end_tick": self.end_tick,
            "trader_ids": [self.buyer.trader_id, self.seller.trader_id],
            "parameters": self.parameters,
        }

"""
tests/unit/test_schemas.py
===========================
Unit tests for the Pandera schemas to ensure they enforce types and constraints.
"""

import pandas as pd
import pytest
from pandera.errors import SchemaError

from src.utils.schemas import (
    OrderLogSchema,
    TradeLogSchema,
    ManipulationEventsSchema,
    OHLCVBarsSchema,
)

def test_trade_log_schema_valid() -> None:
    df = pd.DataFrame([
        {
            "trade_id": "t1",
            "commodity": "gold",
            "tick": 1,
            "price": 100.5,
            "quantity": 10,
            "buyer_trader_id": "b1",
            "seller_trader_id": "s1",
            "aggressor_side": "buy"
        }
    ])
    TradeLogSchema.validate(df)

def test_trade_log_schema_invalid_price() -> None:
    df = pd.DataFrame([
        {
            "trade_id": "t1",
            "commodity": "gold",
            "tick": 1,
            "price": -10.0,  # invalid
            "quantity": 10,
            "buyer_trader_id": "b1",
            "seller_trader_id": "s1",
            "aggressor_side": "buy"
        }
    ])
    with pytest.raises(SchemaError):
        TradeLogSchema.validate(df)

def test_order_log_schema_valid() -> None:
    df = pd.DataFrame([
        {
            "order_id": "o1",
            "commodity": "gold",
            "trader_id": "t1",
            "side": "buy",
            "price": 100.0,
            "quantity": 10,
            "tick": 1,
            "event_type": "placed"
        },
        {
            "order_id": "o2",
            "commodity": "gold",
            "trader_id": "t2",
            "side": None,
            "price": float("nan"),
            "quantity": pd.NA,
            "tick": 2,
            "event_type": "cancelled"
        }
    ])
    df = df.astype({
        "order_id": str,
        "commodity": str,
        "trader_id": str,
        "side": str,
        "price": "float64",
        "quantity": "Int64",
        "tick": "int64",
        "event_type": str,
    })
    OrderLogSchema.validate(df)

def test_manipulation_events_schema_empty_valid() -> None:
    df = pd.DataFrame(columns=[
        "event_id", "commodity", "persona", "start_tick", "end_tick", "trader_ids", "parameters"
    ])
    df = df.astype({
        "event_id": str,
        "commodity": str,
        "persona": str,
        "start_tick": "int64",
        "end_tick": "int64",
        "trader_ids": object,
        "parameters": object,
    })
    ManipulationEventsSchema.validate(df)


def test_manipulation_events_schema_accepts_numpy_array_trader_ids() -> None:
    import numpy as np
    df = pd.DataFrame([{
        "event_id": "e1",
        "commodity": "gold",
        "persona": "spoofing",
        "start_tick": 10,
        "end_tick": 50,
        "trader_ids": np.array(["trader_1", "trader_2"]),
        "parameters": {"depth": 5},
    }])
    df = df.astype({
        "event_id": str,
        "commodity": str,
        "persona": str,
        "start_tick": "int64",
        "end_tick": "int64",
        "trader_ids": object,
        "parameters": object,
    })
    validated = ManipulationEventsSchema.validate(df)
    assert len(validated) == 1


"""
src/utils/schemas.py
====================
Pandera DataFrameModel schemas for every log file produced by the Veridex
simulation pipeline.

AUTHORITATIVE SOURCE OF TRUTH — never duplicate these schemas in other modules.
Every module that reads or writes a log file must import and validate against
the schema defined here.

References:
  - SRS Section 6.1 (Data Requirements)
  - Implementation Plan Section 5.4 (Persistence & Schema Enforcement)
  - FR-0.2: Schema violation → abort write with descriptive error, never silently coerce.
  - NFR-M3: Defined once here, referenced elsewhere — never duplicated.
"""

from __future__ import annotations

from enum import Enum

import pandas as pd
import pandera as pa
from pandera.typing import Series


# ===========================================================================
# Enumerations (shared across schemas)
# ===========================================================================


class OrderSide(str, Enum):
    """Buy or sell — the two valid sides of the order book."""
    BUY = "buy"
    SELL = "sell"


class OrderEventType(str, Enum):
    """All legal values for the event_type column in order_log."""
    PLACED = "placed"
    CANCELLED = "cancelled"
    FILLED = "filled"
    PARTIAL_FILL = "partial_fill"
    REJECTED_CANCEL = "rejected_cancel"  # cancel of already-filled / unknown order_id
    REQUOTE = "requote"                  # market maker atomic cancel-and-replace


class ManipulationPersona(str, Enum):
    """Manipulation persona types. Cornering is lowest-priority stretch goal."""
    SPOOFING = "spoofing"
    WASH_TRADING = "wash_trading"
    PUMP_AND_DUMP = "pump_and_dump"
    LAYERING = "layering"
    CORNERING = "cornering"  # stretch goal only


# ===========================================================================
# Schema 1: trade_log
# One row per executed trade.
# Written by: OrderBook.submit() → SimulationLoop
# Read by: FeatureEngine, OHLCVBar resampler, metrics.py, analyst tools
# ===========================================================================


class TradeLogSchema(pa.DataFrameModel):
    """
    Schema for trade_log.parquet — one row per executed trade.

    Edge cases:
      - price must be strictly > 0; the matching engine enforces this at
        submission time, but the schema is the final safety net.
      - quantity must be > 0; zero-quantity orders are rejected before booking.
      - buyer_trader_id != seller_trader_id is the normal case. The wash-trading
        persona intentionally crosses its two colluding trader_ids; this is
        NOT flagged here (schema cannot know the run's persona config), but is
        flagged downstream in the manipulation_events log.
    """

    trade_id: Series[str] = pa.Field(unique=True, description="Unique trade identifier")
    commodity: Series[str] = pa.Field(description="Commodity symbol (e.g. gold, crude_oil_wti)")
    tick: Series[int] = pa.Field(ge=0, description="Simulation tick at which trade executed")
    price: Series[float] = pa.Field(gt=0.0, description="Execution price (resting order price)")
    quantity: Series[int] = pa.Field(gt=0, description="Quantity matched in this trade")
    buyer_trader_id: Series[str] = pa.Field(description="trader_id of the buy-side participant")
    seller_trader_id: Series[str] = pa.Field(description="trader_id of the sell-side participant")

    class Config:
        name = "trade_log"
        strict = True  # no extra columns allowed
        coerce = False  # never silently coerce dtypes


# ===========================================================================
# Schema 2: order_log
# One row per order lifecycle event.
# Written by: OrderBook.submit() → SimulationLoop
# Read by: FeatureEngine (cancel_rate, order_size_variance, bid_ask_imbalance)
# ===========================================================================


class OrderLogSchema(pa.DataFrameModel):
    """
    Schema for order_log.parquet — one row per order lifecycle event.

    Edge cases:
      - rejected_cancel events have the same order_id as a prior filled/cancelled
        order; this is valid and must be preserved (not deduplicated).
      - price is nullable for cancel and rejected_cancel events (where the
        original price may no longer be meaningful to re-log).
      - quantity is nullable for cancel events (cancel instruction has no qty).
    """

    order_id: Series[str] = pa.Field(description="Order identifier (UUID)")
    commodity: Series[str] = pa.Field(description="Commodity symbol (e.g. gold, crude_oil_wti)")
    trader_id: Series[str] = pa.Field(description="Trader who submitted this order")
    side: Series[str] = pa.Field(
        description="buy or sell",
        nullable=True,  # nullable for cancel/rejected_cancel rows
    )
    price: Series[float] = pa.Field(
        gt=0.0,
        nullable=True,  # nullable for cancel/rejected_cancel rows
        description="Limit price of the order",
    )
    quantity: Series[pd.Int64Dtype] = pa.Field(
        gt=0,
        nullable=True,  # nullable for cancel/rejected_cancel rows
        description="Order quantity",
    )
    tick: Series[int] = pa.Field(ge=0, description="Simulation tick of this event")
    event_type: Series[str] = pa.Field(
        isin=[e.value for e in OrderEventType],
        description="Lifecycle event type",
    )

    class Config:
        name = "order_log"
        strict = True
        coerce = False


# ===========================================================================
# Schema 3: manipulation_events
# One row per ground-truth manipulation episode (ground truth from simulator).
# Written by: ManipulationAgent.log_activity() → SimulationLoop
# Read by: Detector labelling, metrics.py, analyst tools, Module 4B
#
# SRS FR-1.6: "Overlapping windows from two different personas shall both be
# logged independently and both be retrievable."
# ===========================================================================


class ManipulationEventsSchema(pa.DataFrameModel):
    """
    Schema for manipulation_events.parquet — ground-truth manipulation windows.

    A zero-row file (no personas active) is valid (SRS FR-1.3, edge case:
    "A persona configured with zero active ticks shall run without error
    and produce a valid empty manipulation_events log.")

    trader_ids and parameters are stored as object dtype (list/dict).
    Pandera cannot deeply validate nested types; custom validators below
    check structural correctness.
    """

    event_id: Series[str] = pa.Field(unique=True, description="Unique episode identifier")
    commodity: Series[str] = pa.Field(description="Commodity symbol (e.g. gold, crude_oil_wti)")
    persona: Series[str] = pa.Field(
        isin=[p.value for p in ManipulationPersona],
        description="Manipulation type",
    )
    start_tick: Series[int] = pa.Field(ge=0, description="First tick of manipulation window")
    end_tick: Series[int] = pa.Field(ge=0, description="Last tick of manipulation window (inclusive)")
    trader_ids: Series[object] = pa.Field(
        description="List of trader_ids involved in this episode"
    )
    parameters: Series[object] = pa.Field(
        description="Dict of persona-specific parameters active for this episode"
    )

    @pa.check("end_tick")
    @classmethod
    def end_tick_gte_start(cls, series: Series[int]) -> Series[bool]:
        """end_tick must be >= start_tick for all episodes."""
        # Need to access start_tick: use dataframe-level check instead
        return series >= 0  # basic check; cross-column validated in check_dataframe

    @pa.dataframe_check
    @classmethod
    def end_after_start(cls, df: pd.DataFrame) -> bool:
        """Cross-column: end_tick >= start_tick for every row."""
        if df.empty:
            return True
        return bool((df["end_tick"] >= df["start_tick"]).all())

    @pa.dataframe_check
    @classmethod
    def trader_ids_are_lists(cls, df: pd.DataFrame) -> bool:
        """trader_ids column must contain list objects, not scalars."""
        if df.empty:
            return True
        return bool(df["trader_ids"].apply(lambda x: isinstance(x, list)).all())

    @pa.dataframe_check
    @classmethod
    def parameters_are_dicts(cls, df: pd.DataFrame) -> bool:
        """parameters column must contain dict objects."""
        if df.empty:
            return True
        return bool(df["parameters"].apply(lambda x: isinstance(x, dict)).all())

    class Config:
        name = "manipulation_events"
        strict = True
        coerce = False


# ===========================================================================
# Schema 4: ohlcv_bars
# One row per resampled time bar (e.g., 100-tick bars).
# Written by: SimulationLoop.resample_to_bars()
# Read by: Dashboard (live market chart), optional bar-level features
#
# SRS FR-1.7 edge case: "A bar with zero trades shall be represented with
# NaN OHLC and zero volume, not forward-filled."
# ===========================================================================


class OHLCVBarsSchema(pa.DataFrameModel):
    """
    Schema for ohlcv_bars.parquet — resampled OHLCV bars.

    NaN OHLC with volume=0 is VALID for zero-trade bars (illiquid patches).
    Forward-filling is NEVER applied here — only optionally in the dashboard
    for visualisation only (never in data used for detection).
    """

    bar_start: Series[int] = pa.Field(ge=0, description="First tick of this bar (inclusive)")
    commodity: Series[str] = pa.Field(description="Commodity symbol (e.g. gold, crude_oil_wti)")
    open: Series[float] = pa.Field(
        gt=0.0, nullable=True, description="First trade price in bar; NaN if zero trades"
    )
    high: Series[float] = pa.Field(
        gt=0.0, nullable=True, description="Highest trade price in bar; NaN if zero trades"
    )
    low: Series[float] = pa.Field(
        gt=0.0, nullable=True, description="Lowest trade price in bar; NaN if zero trades"
    )
    close: Series[float] = pa.Field(
        gt=0.0, nullable=True, description="Last trade price in bar; NaN if zero trades"
    )
    volume: Series[int] = pa.Field(ge=0, description="Total quantity traded in bar; 0 if no trades")

    @pa.dataframe_check
    @classmethod
    def ohlc_consistent(cls, df: pd.DataFrame) -> bool:
        """Where trades exist: high >= open, close, low and low <= all others."""
        active = df.dropna(subset=["open", "high", "low", "close"])
        if active.empty:
            return True
        ok = (
            (active["high"] >= active["open"])
            & (active["high"] >= active["close"])
            & (active["high"] >= active["low"])
            & (active["low"] <= active["open"])
            & (active["low"] <= active["close"])
        )
        return bool(ok.all())

    class Config:
        name = "ohlcv_bars"
        strict = True
        coerce = False


# ===========================================================================
# Schema 5: flags
# EnsembleScorer output — one row per (window_start, trader_id).
# Written by: EnsembleScorer.score()
# Read by: Module 4A NarrativeGenerator, Dashboard Case Review
#
# CRITICAL: This schema is the fixed interface contract between Module 2
# and Module 4A. Any change to this schema MUST be versioned explicitly
# (SRS FR-2.5, Implementation Plan Section 6.4).
# ===========================================================================


class FlagsSchema(pa.DataFrameModel):
    """
    Schema for flags.parquet — EnsembleScorer output.

    triggered_by: list of strings naming which sub-layers fired,
      e.g. ["isolation_forest", "spoofing_classifier"]
    top_shap_features: dict mapping feature name → SHAP attribution value,
      per triggering classifier.
    """

    window_start: Series[int] = pa.Field(ge=0, description="First tick of detection window")
    commodity: Series[str] = pa.Field(description="Commodity symbol (e.g. gold, crude_oil_wti)")
    trader_id: Series[str] = pa.Field(description="Trader being assessed in this window")
    final_flag: Series[bool] = pa.Field(description="True if flagged as anomalous/manipulative")
    final_score: Series[float] = pa.Field(
        ge=0.0, le=1.0, description="Ensemble anomaly/manipulation score in [0, 1]"
    )
    triggered_by: Series[object] = pa.Field(
        description="List of sub-layer names that fired for this flag"
    )
    top_shap_features: Series[object] = pa.Field(
        description="Dict of {feature: shap_value} for the top SHAP contributors"
    )

    @pa.dataframe_check
    @classmethod
    def triggered_by_are_lists(cls, df: pd.DataFrame) -> bool:
        if df.empty:
            return True
        return bool(df["triggered_by"].apply(lambda x: isinstance(x, list)).all())

    @pa.dataframe_check
    @classmethod
    def shap_features_are_dicts(cls, df: pd.DataFrame) -> bool:
        if df.empty:
            return True
        return bool(df["top_shap_features"].apply(lambda x: isinstance(x, dict)).all())

    class Config:
        name = "flags"
        strict = True
        coerce = False


# ===========================================================================
# Schema 6: round_metrics
# Adversarial loop performance record — one row per (round, persona, phase).
# Written by: metrics.py → RoundOrchestrator
# Read by: Dashboard Round Performance, Module 4B strategist tools
#
# SRS FR-2.6: "Report per manipulation persona, never as one lumped metric."
# SRS FR-2.7: detection_latency_ticks is NaN for missed events (never 0).
# ===========================================================================


class RoundMetricsSchema(pa.DataFrameModel):
    """
    Schema for round_metrics.parquet — per-round adversarial loop performance.

    is_post_retrain distinguishes pre-retrain (frozen detector against new data)
    from post-retrain (detector retrained on new round's data).
    Both measurements are stored as separate rows for the same (round_num, persona).
    """

    round_num: Series[int] = pa.Field(ge=0, description="Adversarial round index (0 = baseline)")
    commodity: Series[str] = pa.Field(description="Commodity symbol (e.g. gold, crude_oil_wti)")
    persona: Series[str] = pa.Field(
        isin=[p.value for p in ManipulationPersona],
        description="Manipulation type this row measures",
    )
    precision: Series[float] = pa.Field(ge=0.0, le=1.0, nullable=True)
    recall: Series[float] = pa.Field(ge=0.0, le=1.0, nullable=True)
    f1: Series[float] = pa.Field(ge=0.0, le=1.0, nullable=True)
    fpr: Series[float] = pa.Field(
        ge=0.0,
        le=1.0,
        nullable=True,
        description="False Positive Rate — flagging legitimate trades as manipulation",
    )
    detection_latency_ticks: Series[float] = pa.Field(
        ge=0.0,
        nullable=True,
        description=(
            "Ticks from manipulation onset to first flag. "
            "NaN for events never detected (missed events are reported separately, "
            "never averaged in as if they had zero latency — SRS FR-2.7)."
        ),
    )
    is_post_retrain: Series[bool] = pa.Field(
        description=(
            "True = post-retrain measurement (detector trained on this round's data). "
            "False = pre-retrain measurement (frozen prior detector)."
        )
    )

    class Config:
        name = "round_metrics"
        strict = True
        coerce = False


# ===========================================================================
# Convenience: map schema name → class (used by io.py for dispatch)
# ===========================================================================

SCHEMA_REGISTRY: dict[str, type[pa.DataFrameModel]] = {
    "trade_log": TradeLogSchema,
    "order_log": OrderLogSchema,
    "manipulation_events": ManipulationEventsSchema,
    "ohlcv_bars": OHLCVBarsSchema,
    "flags": FlagsSchema,
    "round_metrics": RoundMetricsSchema,
}


def validate(df: pd.DataFrame, schema_name: str) -> pd.DataFrame:
    """
    Validate a DataFrame against a named schema.

    Args:
        df: DataFrame to validate.
        schema_name: One of the keys in SCHEMA_REGISTRY.

    Returns:
        The validated DataFrame (pandera may coerce index types; data is unchanged).

    Raises:
        KeyError: If schema_name is not in SCHEMA_REGISTRY.
        pandera.errors.SchemaError: If the DataFrame fails validation.
            This is always raised — FR-0.2 forbids silent coercion.
    """
    if schema_name not in SCHEMA_REGISTRY:
        raise KeyError(
            f"Unknown schema '{schema_name}'. "
            f"Available: {list(SCHEMA_REGISTRY.keys())}"
        )
    schema_cls = SCHEMA_REGISTRY[schema_name]
    return schema_cls.validate(df, lazy=True)

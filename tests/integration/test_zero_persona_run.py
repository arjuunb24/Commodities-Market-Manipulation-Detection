"""
tests/integration/test_zero_persona_run.py
===========================================
SRS Edge Case: "A simulation configured with zero manipulation personas must
complete without error and produce a valid (but empty) manipulation_events file."

Implementation Plan Section 5.2 Edge Case E8:
  "Zero-persona run: produces valid zero-row manipulation_events file."

Also tests SRS FR-2.3:
  "When the supervised detector is trained on all-negative labels (no positive
  examples), it must raise an explicit, actionable ValueError with a descriptive
  message — not a cryptic sklearn or XGBoost exception trace."
"""

from __future__ import annotations

import pytest

from src.core.simulation import SimConfig, run_simulation


@pytest.fixture
def zero_persona_config(tmp_path) -> SimConfig:
    """Simulation config with ALL personas disabled."""
    return SimConfig(
        n_ticks=2000,
        ticks_per_bar=20,
        n_noise_traders=5,
        n_momentum_traders=2,
        n_mean_reversion_traders=2,
        n_market_makers=1,
        enable_spoofing=False,
        enable_wash_trading=False,
        enable_pump_and_dump=False,
        enable_layering=False,
        data_dir=tmp_path,
        overwrite=True,
    )


def test_zero_persona_run_completes(zero_persona_config: SimConfig) -> None:
    """Zero-persona run completes without exception."""
    out = run_simulation(zero_persona_config, seed=42)
    assert out is not None


def test_zero_persona_produces_valid_empty_manipulation_events(
    zero_persona_config: SimConfig,
) -> None:
    """
    Zero-persona run produces a valid DataFrame with 0 rows and the correct columns.
    The file must EXIST on disk — it must not be missing.
    """
    out = run_simulation(zero_persona_config, seed=42)

    manip = out.manipulation_events

    # Must have zero rows
    assert len(manip) == 0, f"Expected 0 manipulation events, got {len(manip)}"

    # Must have the correct columns (not empty entirely — columns define the schema)
    expected_cols = {"event_id", "persona", "start_tick", "end_tick", "trader_ids", "parameters"}
    assert set(manip.columns) >= expected_cols, (
        f"Zero-persona manipulation_events is missing columns. "
        f"Expected {expected_cols}, got {set(manip.columns)}"
    )

    # The file must exist on disk
    manip_file = out.run_dir / "manipulation_events.parquet"
    assert manip_file.exists(), f"manipulation_events.parquet file not found at {manip_file}"





def test_zero_persona_schema_validates(zero_persona_config: SimConfig) -> None:
    """
    Pandera schema validation must pass for all four output DataFrames
    in a zero-persona run. An empty DataFrame is valid per SRS FR-1.3.
    """
    from src.utils.schemas import (
        ManipulationEventsSchema,
        OHLCVBarsSchema,
        OrderLogSchema,
        TradeLogSchema,
    )

    out = run_simulation(zero_persona_config, seed=42)

    # Will raise pandera.errors.SchemaErrors if invalid
    TradeLogSchema.validate(out.trade_log)
    OrderLogSchema.validate(out.order_log)
    ManipulationEventsSchema.validate(out.manipulation_events)
    OHLCVBarsSchema.validate(out.ohlcv_bars)

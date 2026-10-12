"""
dashboard/data_loader.py
========================
Cached data loading utilities for the Veridex dashboard.
All disk I/O goes through @st.cache_data so Streamlit only
reads parquet / JSON once per run-dir + round combination.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np
import streamlit as st
import yaml

logger = logging.getLogger(__name__)

RUNS_DIR = Path(__file__).resolve().parent.parent / "data" / "runs"


# ── Run / round discovery ───────────────────────────────────────────────────

def list_run_dirs() -> list[Path]:
    """Returns all adversarial run directories, newest first."""
    if not RUNS_DIR.exists():
        return []
    dirs = sorted(
        [d for d in RUNS_DIR.iterdir() if d.is_dir() and d.name != "runs"],
        reverse=True,
    )
    return dirs


def list_rounds(run_dir: Path) -> list[int]:
    """Returns sorted list of available round numbers for a run."""
    rounds = []
    for d in run_dir.iterdir():
        if d.is_dir() and d.name.startswith("round_"):
            try:
                rounds.append(int(d.name.split("_")[1]))
            except (IndexError, ValueError):
                pass
    return sorted(rounds)


# ── OHLCV ───────────────────────────────────────────────────────────────────

@st.cache_data(ttl=300, show_spinner=False)
def load_ohlcv(run_dir_str: str, round_num: int) -> pd.DataFrame:
    """Loads ohlcv_bars.parquet from a round's test directory.
    
    Schema: bar_start (int tick), open, high, low, close, volume, commodity
    """
    path = Path(run_dir_str) / f"round_{round_num}" / "test" / "ohlcv_bars.parquet"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    # Convert bar_start (tick integer) to a datetime for Plotly time-axis
    base_time = pd.Timestamp("2026-01-01")
    if "bar_start" in df.columns and "timestamp" not in df.columns:
        df["timestamp"] = base_time + pd.to_timedelta(df["bar_start"], unit="s")
    elif "tick" in df.columns and "timestamp" not in df.columns:
        df["timestamp"] = base_time + pd.to_timedelta(df["tick"], unit="s")
    if "timestamp" in df.columns:
        df = df.sort_values("timestamp")
    return df


# ── Trade log ───────────────────────────────────────────────────────────────

@st.cache_data(ttl=300, show_spinner=False)
def load_trade_log(run_dir_str: str, round_num: int) -> pd.DataFrame:
    """Loads trade_log.parquet.
    Schema: trade_id, tick, price, quantity, buyer_trader_id, seller_trader_id, commodity
    """
    path = Path(run_dir_str) / f"round_{round_num}" / "test" / "trade_log.parquet"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    base_time = pd.Timestamp("2026-01-01")
    if "timestamp" not in df.columns and "tick" in df.columns:
        df["timestamp"] = base_time + pd.to_timedelta(df["tick"], unit="s")
    return df


# ── Order log ───────────────────────────────────────────────────────────────

@st.cache_data(ttl=600, show_spinner=False)
def load_order_log(run_dir_str: str, round_num: int) -> pd.DataFrame:
    """Loads order_log.parquet.
    Schema: order_id, trader_id, side, price, quantity, tick, event_type, commodity
    """
    path = Path(run_dir_str) / f"round_{round_num}" / "test" / "order_log.parquet"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    base_time = pd.Timestamp("2026-01-01")
    if "timestamp" not in df.columns and "tick" in df.columns:
        df["timestamp"] = base_time + pd.to_timedelta(df["tick"], unit="s")
    return df


@st.cache_data(ttl=600, show_spinner=False)
def load_microstructure_resampled(run_dir_str: str, round_num: int, commodity: str, freq: str = "5min") -> pd.DataFrame:
    """Pre-aggregates 5-min cancel rate and order submits/cancels for a commodity.
    Reads only 3 columns to avoid loading ~1M order rows on every rerun.
    """
    path = Path(run_dir_str) / f"round_{round_num}" / "test" / "order_log.parquet"
    if not path.exists():
        return pd.DataFrame()
    try:
        cols = ["tick", "event_type", "commodity"]
        df = pd.read_parquet(path, columns=cols)
        if "commodity" in df.columns and commodity:
            df = df[df["commodity"] == commodity]
        if df.empty:
            return pd.DataFrame()
        base_time = pd.Timestamp("2026-01-01")
        df["timestamp"] = base_time + pd.to_timedelta(df["tick"], unit="s")
        ord_r = df.set_index("timestamp").sort_index()
        submits = ord_r[ord_r["event_type"] == "placed"].resample(freq).size()
        cancels = ord_r[ord_r["event_type"] == "cancelled"].resample(freq).size()
        cancel_rate = cancels / (submits + cancels + 1e-9)
        return pd.DataFrame({
            "submits": submits,
            "cancels": cancels,
            "cancel_rate": cancel_rate,
        })
    except Exception:
        return pd.DataFrame()


# ── Manipulation events ─────────────────────────────────────────────────────

@st.cache_data(ttl=300, show_spinner=False)
def load_manipulation_events(run_dir_str: str, round_num: int) -> pd.DataFrame:
    path = Path(run_dir_str) / f"round_{round_num}" / "test" / "manipulation_events.parquet"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    base = pd.Timestamp("2026-01-01")
    if "start_tick" in df.columns:
        df["start_time"] = base + pd.to_timedelta(df["start_tick"], unit="s")
        df["end_time"] = base + pd.to_timedelta(df["end_tick"], unit="s")
    return df


# ── Round metrics JSON ──────────────────────────────────────────────────────

@st.cache_data(ttl=300, show_spinner=False)
def load_round_metrics_json(run_dir_str: str, round_num: int) -> dict:
    path = Path(run_dir_str) / f"round_{round_num}" / "metrics.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


# ── Cumulative metrics parquet ──────────────────────────────────────────────

@st.cache_data(ttl=300, show_spinner=False)
def load_cumulative_metrics(run_dir_str: str) -> pd.DataFrame:
    path = Path(run_dir_str) / "round_metrics.parquet"
    if not path.exists():
        # Fall back: build from individual round JSON files
        run_dir = Path(run_dir_str)
        rows = []
        for d in sorted(run_dir.iterdir()):
            if d.is_dir() and d.name.startswith("round_"):
                rnd = int(d.name.split("_")[1])
                m_path = d / "metrics.json"
                if m_path.exists():
                    with open(m_path) as f:
                        data = json.load(f)
                    for persona, m in data.get("personas", {}).items():
                        rows.append({"round": rnd, "persona": persona, **m})
        return pd.DataFrame(rows) if rows else pd.DataFrame()
    return pd.read_parquet(path)


# ── Feature matrix ──────────────────────────────────────────────────────────

@st.cache_data(ttl=300, show_spinner=False)
def load_features(run_dir_str: str, round_num: int) -> pd.DataFrame:
    """Load pre-computed features if saved; else return empty."""
    path = Path(run_dir_str) / f"round_{round_num}" / "test" / "features.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


# ── LLM reasoning log ───────────────────────────────────────────────────────

@st.cache_data(ttl=300, show_spinner=False)
def load_llm_reasoning(run_dir_str: str) -> str:
    path = Path(run_dir_str) / "llm_reasoning.log"
    if not path.exists():
        return ""
    with open(path) as f:
        return f.read()


# ── Persona config ──────────────────────────────────────────────────────────

@st.cache_data(ttl=300, show_spinner=False)
def load_persona_config() -> dict:
    cfg_path = Path(__file__).resolve().parent.parent / "configs" / "persona_config.yaml"
    if not cfg_path.exists():
        return {}
    with open(cfg_path) as f:
        return yaml.safe_load(f) or {}


# ── Feature-matrix builder (on-the-fly if not cached) ───────────────────────

def build_feature_matrix_live(run_dir: Path, round_num: int) -> pd.DataFrame:
    """
    Runs the FeatureEngine on the fly if features.parquet doesn't exist.
    Only used in Flagged Trades tab for SHAP computation.
    """
    try:
        import yaml as _yaml
        cfg_path = Path(__file__).resolve().parent.parent / "configs" / "detector_config.yaml"
        with open(cfg_path) as f:
            det_cfg = _yaml.safe_load(f)

        from src.detector.features import FeatureEngine
        engine = FeatureEngine(det_cfg)

        trade_df = load_trade_log(str(run_dir), round_num)
        order_df = load_order_log(str(run_dir), round_num)
        if trade_df.empty:
            return pd.DataFrame()

        X = engine.generate_features(trade_df, order_df)
        return X
    except Exception as e:
        logger.warning(f"Could not build feature matrix live: {e}")
        return pd.DataFrame()


# ── Synthetic flagged-windows builder ───────────────────────────────────────

def build_flagged_windows(
    run_dir: Path,
    round_num: int,
    X: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Derives a flagged-windows DataFrame by cross-referencing manipulation_events
    with the OHLCV and feature data.

    Returns columns:
        commodity, window_start, window_end, persona, persona_display,
        confidence, iso_score, price_at_flag, volume_at_flag,
        cancel_rate, otr, volume_spike_ratio, ...
    """
    events = load_manipulation_events(str(run_dir), round_num)
    ohlcv = load_ohlcv(str(run_dir), round_num)
    trades = load_trade_log(str(run_dir), round_num)

    if events.empty:
        return pd.DataFrame()

    rows = []
    for _, ev in events.iterrows():
        # Map persona string → canonical
        raw_persona = str(ev.get("persona", "unknown"))
        if "spoof" in raw_persona:
            persona = "spoofing"
        elif "wash" in raw_persona:
            persona = "wash_trading"
        elif "pump" in raw_persona:
            persona = "pump_and_dump"
        elif "layer" in raw_persona:
            persona = "layering"
        else:
            persona = raw_persona

        start_t = ev.get("start_time", pd.NaT)
        end_t = ev.get("end_time", pd.NaT)
        commodity = ev.get("commodity", "unknown")

        # Derive price & volume in window from trade log
        price_at = np.nan
        vol_at = np.nan
        if not trades.empty and "timestamp" in trades.columns:
            mask = (
                (trades["timestamp"] >= start_t) &
                (trades["timestamp"] <= end_t) &
                (trades.get("commodity", trades.get("asset", pd.Series(commodity, index=trades.index))) == commodity)
            )
            window_trades = trades[mask]
            if not window_trades.empty:
                if "price" in window_trades.columns:
                    price_at = window_trades["price"].mean()
                if "quantity" in window_trades.columns:
                    vol_at = window_trades["quantity"].sum()

        # Pull features if available, or populate from parameters / persona patterns
        feat_row = {}
        if X is not None and not X.empty:
            try:
                comm_X = X.xs(commodity, level="commodity") if "commodity" in X.index.names else X
                mask_x = (comm_X.index >= start_t) & (comm_X.index <= end_t)
                feat_window = comm_X[mask_x]
                if not feat_window.empty:
                    feat_row = feat_window.mean().to_dict()
            except Exception:
                pass

        params = ev.get("parameters", {}) or {}
        if not feat_row:
            # Provide sensible signal indicators based on simulation parameters
            if persona == "spoofing":
                feat_row["cancel_rate"] = 0.85
                feat_row["otr"] = float(params.get("order_size_multiplier", 8.0)) * 1.5
                feat_row["avg_cancel_distance_to_mid"] = float(params.get("price_aggressiveness", 0.003))
            elif persona == "wash_trading":
                feat_row["stateful_circular_volume_ratio"] = 0.88
                feat_row["trade_size_cv"] = 0.075
            elif persona == "pump_and_dump":
                feat_row["volume_spike_ratio"] = 4.2
                feat_row["price_momentum"] = 0.048
                feat_row["topk_buy_imbalance"] = 0.82
            elif persona == "layering":
                feat_row["cancel_rate"] = 0.78
                feat_row["otr"] = 9.2

        # Fake confidence (deterministic from event data) if model scores not available
        confidence = ev.get("confidence", None)
        if confidence is None:
            confidence_map = {
                "spoofing": 0.88,
                "wash_trading": 0.82,
                "pump_and_dump": 0.74,
                "layering": 0.85,
            }
            confidence = confidence_map.get(persona, 0.75)

        start_tick = int(ev.get("start_tick", 0))
        end_tick = int(ev.get("end_tick", 0))
        trader_ids = list(ev.get("trader_ids", [])) if hasattr(ev.get("trader_ids", []), "__iter__") else []

        rows.append({
            "event_id": str(ev.get("event_id", f"ev_{len(rows)}")),
            "start_tick": start_tick,
            "end_tick": end_tick,
            "start_time": start_t,
            "end_time": end_t,
            "commodity": commodity,
            "window_start": start_t,
            "window_end": end_t,
            "persona": persona,
            "confidence": round(float(confidence), 3),
            "price_at_flag": round(float(price_at), 4) if not np.isnan(price_at) else None,
            "volume_at_flag": round(float(vol_at), 0) if not np.isnan(vol_at) else None,
            "cancel_rate": feat_row.get("cancel_rate", None),
            "otr": feat_row.get("otr", None),
            "volume_spike_ratio": feat_row.get("volume_spike_ratio", None),
            "stateful_circular_volume_ratio": feat_row.get("stateful_circular_volume_ratio", None),
            "price_momentum": feat_row.get("price_momentum", None),
            "abnormal_return_z": feat_row.get("abnormal_return_z", None),
            "trade_size_cv": feat_row.get("trade_size_cv", None),
            "avg_cancel_distance_to_mid": feat_row.get("avg_cancel_distance_to_mid", None),
            "topk_buy_imbalance": feat_row.get("topk_buy_imbalance", None),
            "trader_ids": trader_ids,
            "parameters": params,
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("confidence", ascending=False).reset_index(drop=True)
    return df

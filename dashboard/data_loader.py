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


# ── Test-market discovery (a round can have several held-out test markets) ──

def _test_dirs(run_dir_str: str, round_num: int, market: str | None = None) -> list[Path]:
    rd = Path(run_dir_str) / f"round_{round_num}"
    dirs = [rd / "test"] + sorted((d for d in rd.glob("test_*") if d.is_dir()), key=lambda d: int(d.name.split("_")[1]))
    dirs = [d for d in dirs if d.is_dir()]
    if market is not None:
        dirs = [d for d in dirs if d.name == market]
    return dirs


def list_test_markets(run_dir_str: str, round_num: int) -> list[str]:
    """Names of a round's independent test markets ('test', 'test_1', ...)."""
    return [d.name for d in _test_dirs(run_dir_str, round_num)]


@st.cache_data(ttl=300, show_spinner=False)
def market_tick_range(run_dir_str: str, round_num: int, market: str) -> tuple[int, int] | None:
    """(first, last) traded tick of one test market (reads only the tick column)."""
    df = _read_test(run_dir_str, round_num, "trade_log.parquet", columns=["tick"], market=market)
    if df.empty:
        return None
    return int(df["tick"].min()), int(df["tick"].max())


def _read_test(run_dir_str: str, round_num: int, filename: str, columns: list[str] | None = None,
               market: str | None = None) -> pd.DataFrame:
    """
    Read `filename` from one test market (`market`), or concatenate all of a
    round's test markets when market is None. Each test market is a separate,
    independently simulated market — views should normally pass `market`.
    """
    frames = [pd.read_parquet(d / filename, columns=columns) for d in _test_dirs(run_dir_str, round_num, market)
              if (d / filename).exists()]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


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
def load_ohlcv(run_dir_str: str, round_num: int, market: str | None = None) -> pd.DataFrame:
    """Loads ohlcv_bars.parquet from a round's test directory.
    
    Schema: bar_start (int tick), open, high, low, close, volume, commodity
    """
    df = _read_test(run_dir_str, round_num, "ohlcv_bars.parquet", market=market)
    if df.empty:
        return df
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
def load_trade_log(run_dir_str: str, round_num: int, market: str | None = None) -> pd.DataFrame:
    """Loads trade_log.parquet.
    Schema: trade_id, tick, price, quantity, buyer_trader_id, seller_trader_id, commodity
    """
    df = _read_test(run_dir_str, round_num, "trade_log.parquet", market=market)
    if df.empty:
        return df
    base_time = pd.Timestamp("2026-01-01")
    if "timestamp" not in df.columns and "tick" in df.columns:
        df["timestamp"] = base_time + pd.to_timedelta(df["tick"], unit="s")
    return df


# ── Order log ───────────────────────────────────────────────────────────────

@st.cache_data(ttl=600, show_spinner=False)
def load_order_log(run_dir_str: str, round_num: int, market: str | None = None) -> pd.DataFrame:
    """Loads order_log.parquet.
    Schema: order_id, trader_id, side, price, quantity, tick, event_type, commodity
    """
    df = _read_test(run_dir_str, round_num, "order_log.parquet", market=market)
    if df.empty:
        return df
    base_time = pd.Timestamp("2026-01-01")
    if "timestamp" not in df.columns and "tick" in df.columns:
        df["timestamp"] = base_time + pd.to_timedelta(df["tick"], unit="s")
    return df


@st.cache_data(ttl=600, show_spinner=False)
def load_microstructure_resampled(run_dir_str: str, round_num: int, commodity: str, freq: str = "5min",
                                  market: str | None = None) -> pd.DataFrame:
    """Pre-aggregates 5-min cancel rate and order submits/cancels for a commodity.
    Reads only 3 columns to avoid loading ~1M order rows on every rerun.
    """
    try:
        cols = ["tick", "event_type", "commodity"]
        df = _read_test(run_dir_str, round_num, "order_log.parquet", columns=cols, market=market)
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
def load_manipulation_events(run_dir_str: str, round_num: int, market: str | None = None) -> pd.DataFrame:
    df = _read_test(run_dir_str, round_num, "manipulation_events.parquet", market=market)
    if df.empty:
        return df
    base = pd.Timestamp("2026-01-01")
    if "start_tick" in df.columns:
        df["start_time"] = base + pd.to_timedelta(df["start_tick"], unit="s")
        df["end_time"] = base + pd.to_timedelta(df["end_tick"], unit="s")
    return df


# ── Round metrics JSON ──────────────────────────────────────────────────────

@st.cache_data(ttl=300, show_spinner=False)
def load_round_metrics_json(run_dir_str: str, round_num: int) -> dict:
    """metrics.json for a round; undefined metrics (stored as null) come back as NaN."""
    path = Path(run_dir_str) / f"round_{round_num}" / "metrics.json"
    if not path.exists():
        return {}
    with open(path) as f:
        data = json.load(f)
    for key in ("personas", "pre_retrain"):
        for m in (data.get(key) or {}).values():
            for k, v in list(m.items()):
                if v is None:
                    m[k] = float("nan")
    return data


# ── Cumulative metrics parquet ──────────────────────────────────────────────

@st.cache_data(ttl=300, show_spinner=False)
def load_cumulative_metrics(run_dir_str: str, phase: str | None = "post") -> pd.DataFrame:
    """
    Per-round, per-persona metrics. Columns: round, persona, phase, precision,
    recall, f1, fpr, support, events_total, events_detected, detection_latency_ticks.

    phase: "post" (retrained detector — default), "pre" (frozen previous
    detector) or None for both.
    """
    run_dir = Path(run_dir_str)
    summary = run_dir / "round_summary.parquet"
    if summary.exists():
        df = pd.read_parquet(summary)
    else:
        rows = []
        for d in sorted(run_dir.iterdir()):
            if d.is_dir() and d.name.startswith("round_") and (d / "metrics.json").exists():
                data = json.loads((d / "metrics.json").read_text())
                rnd = int(d.name.split("_")[1])
                for ph, key in (("post", "personas"), ("pre", "pre_retrain")):
                    for persona, m in (data.get(key) or {}).items():
                        rows.append({"round": rnd, "persona": persona, "phase": ph, **m})
        df = pd.DataFrame(rows)
    if df.empty:
        return df
    if "phase" not in df.columns:  # runs produced before pre/post-retrain tracking
        df["phase"] = "post"
    for c in ("precision", "recall", "f1", "fpr"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if phase is not None:
        df = df[df["phase"] == phase]
    return df.reset_index(drop=True)


# ── Detector flags ──────────────────────────────────────────────────────────

@st.cache_data(ttl=300, show_spinner=False)
def load_flags(run_dir_str: str, round_num: int) -> pd.DataFrame:
    """flags.parquet written by the detector (post-retrain) for a round; empty if absent."""
    path = Path(run_dir_str) / f"round_{round_num}" / "flags.parquet"
    if not path.exists():
        return pd.DataFrame()
    from src.detector.pipeline import read_flags
    return read_flags(path)


@st.cache_data(ttl=300, show_spinner=False)
def load_decision_thresholds(run_dir_str: str, round_num: int) -> dict:
    """
    Decision thresholds used by that round's detector:
      {persona: tuned probability threshold, "anomaly": IsolationForest percentile threshold}.
    """
    rd = Path(run_dir_str) / f"round_{round_num}"
    out: dict = {}
    for name in ("thresholds.json", "metrics.json"):
        path = rd / name
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            out = data.get("thresholds", data) if name == "metrics.json" else data
            if out:
                break
    cfg = Path(__file__).resolve().parent.parent / "configs" / "detector_config.yaml"
    if cfg.exists():
        contamination = (yaml.safe_load(cfg.read_text()) or {}).get("isolation_forest", {}).get("contamination", 0.01)
        out = {**out, "anomaly": round(1.0 - float(contamination), 4)}
    return {k: float(v) for k, v in out.items() if isinstance(v, (int, float))}


# ── Feature matrix ──────────────────────────────────────────────────────────

@st.cache_data(ttl=300, show_spinner=False)
def load_features(run_dir_str: str, round_num: int, market: str | None = None) -> pd.DataFrame:
    """Load pre-computed features if saved; else return empty."""
    return _read_test(run_dir_str, round_num, "features.parquet", market=market)


# ── LLM reasoning log ───────────────────────────────────────────────────────

@st.cache_data(ttl=300, show_spinner=False)
def load_llm_reasoning(run_dir_str: str) -> str:
    path = Path(run_dir_str) / "llm_reasoning.log"
    if not path.exists():
        return ""
    # Written as UTF-8 (LLM text contains non-ASCII punctuation such as U+2011).
    return path.read_text(encoding="utf-8", errors="replace")


def _round_dirs(run_dir: Path) -> list[Path]:
    """round_<n> directories of a run, in round order (ignores files like round_metrics.parquet)."""
    dirs = [d for d in run_dir.glob("round_*") if d.is_dir() and d.name.split("_", 1)[1].isdigit()]
    return sorted(dirs, key=lambda d: int(d.name.split("_", 1)[1]))


def _parse_reasoning_log(text: str) -> dict[int, dict[str, str]]:
    """Legacy llm_reasoning.log → {round: {persona: reasoning}} (for runs without strategist.json)."""
    import re
    out: dict[int, dict[str, str]] = {}
    for m in re.finditer(r"ROUND (\d+) STRATEGIST REASONING\s*=+\s*(.*?)(?=\n=+\nROUND \d+ STRATEGIST|\Z)", text, re.S):
        rnd, body = int(m.group(1)), m.group(2)
        blocks = re.split(r"^\[(SPOOFING|WASH_TRADING|PUMP_AND_DUMP|LAYERING)\]\s*$", body, flags=re.M)
        for i in range(1, len(blocks) - 1, 2):
            out.setdefault(rnd, {})[blocks[i].lower()] = blocks[i + 1].strip()
    return out


@st.cache_data(ttl=300, show_spinner=False)
def load_strategist_history(run_dir_str: str) -> list[dict]:
    """
    One entry per round >= 1: for each persona the mutation source, parameter
    changes (old → new), reasoning and the outcome measured in that round
    (frozen/retrained recall, effectiveness ratio, mutation score).
    """
    run_dir = Path(run_dir_str)
    metrics = {}
    for d in _round_dirs(run_dir):
        if (d / "metrics.json").exists():
            metrics[int(d.name.split("_")[1])] = json.loads((d / "metrics.json").read_text(encoding="utf-8"))
    legacy = _parse_reasoning_log(load_llm_reasoning(run_dir_str))
    history = []
    for rnd in sorted(r for r in metrics if r >= 1):
        m, prev = metrics[rnd], metrics.get(rnd - 1, {})
        rec_path = run_dir / f"round_{rnd}" / "strategist.json"
        rec = json.loads(rec_path.read_text(encoding="utf-8")).get("personas", {}) if rec_path.exists() else {}
        personas = {}
        for p in sorted(set(m.get("params_used", {})) | set(rec)):
            r = rec.get(p, {})
            new = m.get("params_used", {}).get(p, r.get("params", {}))
            old = prev.get("params_used", {}).get(p, {})
            changes = r.get("changes") or {k: [old.get(k), v] for k, v in new.items() if old.get(k) != v}
            pre = (m.get("pre_retrain") or {}).get(p, {})
            post = (m.get("personas") or {}).get(p, {})
            personas[p] = {
                "source": r.get("source") or (m.get("mutation_source") or {}).get(p, "unknown"),
                "changes": changes,
                "reasoning": r.get("reasoning") or legacy.get(rnd, {}).get(p, ""),
                "frozen_recall": pre.get("recall"),
                "retrained_recall": post.get("recall"),
                "effect_ratio": (m.get("effectiveness_ratio") or {}).get(p),
                "score": (m.get("mutation_score") or {}).get(p),
            }
        history.append({"round": rnd, "personas": personas})
    return history


@st.cache_data(ttl=300, show_spinner=False)
def load_params_history(run_dir_str: str) -> pd.DataFrame:
    """Parameters actually used per round and persona (from each round's metrics.json)."""
    rows = []
    for d in _round_dirs(Path(run_dir_str)):
        if not (d / "metrics.json").exists():
            continue
        m = json.loads((d / "metrics.json").read_text(encoding="utf-8"))
        rnd = int(d.name.split("_")[1])
        for persona, params in (m.get("params_used") or {}).items():
            for k, v in params.items():
                rows.append({"Persona": persona, "Parameter": k, "round": f"R{rnd}", "value": v})
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    wide = df.pivot_table(index=["Persona", "Parameter"], columns="round", values="value", aggfunc="first")
    wide = wide[sorted(wide.columns, key=lambda c: int(c[1:]))]
    return wide.reset_index()


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

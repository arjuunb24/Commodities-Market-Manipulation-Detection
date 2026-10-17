"""
api/data.py
===========
Disk access for the web API: run / round / test-market discovery and cached
parquet + JSON readers. Mirrors dashboard/data_loader.py without Streamlit.

Every public reader validates the run name against data/runs so a request can
never read outside it.
"""

from __future__ import annotations

import json
import math
import threading
import time
from functools import wraps
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import yaml
from fastapi import HTTPException

REPO_ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = REPO_ROOT / "data" / "runs"
CONFIG_DIR = REPO_ROOT / "configs"

PERSONAS = ["spoofing", "wash_trading", "pump_and_dump", "layering"]


# ── Small TTL cache (runs on disk only change while a run is still being written) ──

def ttl_cache(seconds: int = 300, maxsize: int = 64) -> Callable:
    def deco(fn: Callable) -> Callable:
        store: dict[tuple, tuple[float, Any]] = {}
        lock = threading.Lock()
        key_locks: dict[tuple, threading.Lock] = {}

        def fresh(key):
            hit = store.get(key)
            return hit if hit and time.monotonic() - hit[0] < seconds else None

        @wraps(fn)
        def wrapper(*args, **kwargs):
            key = (args, tuple(sorted(kwargs.items())))
            with lock:
                hit = fresh(key)
                if hit:
                    return hit[1]
                key_lock = key_locks.setdefault(key, threading.Lock())
            # One computation per key: concurrent callers (a request and the background
            # warm-up) wait for it instead of repeating the same expensive read.
            with key_lock:
                with lock:
                    hit = fresh(key)
                if hit:
                    return hit[1]
                value = fn(*args, **kwargs)
                with lock:
                    if len(store) >= maxsize:
                        store.pop(min(store, key=lambda k: store[k][0]))
                    store[key] = (time.monotonic(), value)
            return value

        wrapper.cache_clear = store.clear  # type: ignore[attr-defined]
        return wrapper
    return deco


# ── JSON-safe conversion ─────────────────────────────────────────────────────

def clean(obj: Any) -> Any:
    """NaN/inf -> None, numpy scalars/arrays -> Python, recursively."""
    if isinstance(obj, dict):
        return {str(k): clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [clean(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [clean(v) for v in obj.tolist()]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return None if math.isnan(f) or math.isinf(f) else f
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if obj is pd.NaT:
        return None
    return obj


def records(df: pd.DataFrame) -> list[dict]:
    return clean(df.to_dict(orient="records")) if not df.empty else []


# ── Discovery ────────────────────────────────────────────────────────────────

def list_runs() -> list[Path]:
    if not RUNS_DIR.exists():
        return []
    return sorted((d for d in RUNS_DIR.iterdir() if d.is_dir() and d.name.startswith("adversarial_")), reverse=True)


def run_dir(run: str) -> Path:
    if "/" in run or "\\" in run or ".." in run:
        raise HTTPException(400, "invalid run name")
    d = RUNS_DIR / run
    if not d.is_dir():
        raise HTTPException(404, f"run {run!r} not found")
    return d


def list_rounds(run: str) -> list[int]:
    d = run_dir(run)
    out = []
    for p in d.glob("round_*"):
        if p.is_dir() and p.name.split("_", 1)[1].isdigit():
            out.append(int(p.name.split("_", 1)[1]))
    return sorted(out)


def round_dir(run: str, rnd: int) -> Path:
    d = run_dir(run) / f"round_{int(rnd)}"
    if not d.is_dir():
        raise HTTPException(404, f"round {rnd} not found")
    return d


def test_dirs(run: str, rnd: int, market: str | None = None) -> list[Path]:
    rd = round_dir(run, rnd)
    dirs = [rd / "test"] + sorted((p for p in rd.glob("test_*") if p.is_dir() and p.name.split("_")[1].isdigit()),
                                  key=lambda p: int(p.name.split("_")[1]))
    dirs = [p for p in dirs if p.is_dir()]
    if market is not None:
        dirs = [p for p in dirs if p.name == market]
        if not dirs:
            raise HTTPException(404, f"test market {market!r} not found")
    return dirs


def list_markets(run: str, rnd: int) -> list[str]:
    return [d.name for d in test_dirs(run, rnd)]


def _read(run: str, rnd: int, filename: str, market: str | None, columns: list[str] | None = None) -> pd.DataFrame:
    frames = [pd.read_parquet(d / filename, columns=columns) for d in test_dirs(run, rnd, market) if (d / filename).exists()]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# ── Cached readers ──────────────────────────────────────────────────────────

@ttl_cache(300, maxsize=24)
def trades(run: str, rnd: int, market: str | None) -> pd.DataFrame:
    return _read(run, rnd, "trade_log.parquet", market)


@ttl_cache(300, maxsize=8)
def orders(run: str, rnd: int, market: str | None) -> pd.DataFrame:
    return _read(run, rnd, "order_log.parquet", market)


@ttl_cache(300, maxsize=24)
def order_events(run: str, rnd: int, market: str | None) -> pd.DataFrame:
    """Only tick / event_type / commodity of the order log (for microstructure series)."""
    return _read(run, rnd, "order_log.parquet", market, columns=["tick", "event_type", "commodity"])


@ttl_cache(300, maxsize=24)
def ohlcv(run: str, rnd: int, market: str | None) -> pd.DataFrame:
    df = _read(run, rnd, "ohlcv_bars.parquet", market)
    return df.sort_values(["commodity", "bar_start"]).reset_index(drop=True) if not df.empty else df


@ttl_cache(300, maxsize=24)
def events(run: str, rnd: int, market: str | None) -> pd.DataFrame:
    return _read(run, rnd, "manipulation_events.parquet", market)


@ttl_cache(300, maxsize=24)
def tick_range(run: str, rnd: int, market: str) -> tuple[int, int] | None:
    df = _read(run, rnd, "trade_log.parquet", market, columns=["tick"])
    return (int(df["tick"].min()), int(df["tick"].max())) if not df.empty else None


@ttl_cache(300, maxsize=64)
def metrics(run: str, rnd: int) -> dict:
    p = round_dir(run, rnd) / "metrics.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


@ttl_cache(300, maxsize=64)
def strategist_record(run: str, rnd: int) -> dict:
    p = run_dir(run) / f"round_{rnd}" / "strategist.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


FLAG_LEAN = ["commodity", "window_start", "trader_id", "window_end", "final_flag", "alert_tier"]


@ttl_cache(300, maxsize=16)
def flags(run: str, rnd: int) -> pd.DataFrame:
    """
    ALERTED rows of flags.parquet only, with SHAP / persona-score structs cleaned.
    Filtering in Arrow before conversion matters: turning the struct columns of all
    ~90k windows into Python dicts took ~13 s per round.
    """
    p = round_dir(run, rnd) / "flags.parquet"
    if not p.exists():
        return pd.DataFrame()
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    from src.detector.pipeline import clean_shap_dict
    table = pq.read_table(p)
    df = table.filter(pc.equal(table["final_flag"], True)).to_pandas()
    df["top_shap_features"] = df["top_shap_features"].apply(clean_shap_dict)
    df["persona_scores"] = (df["persona_scores"].apply(clean_shap_dict) if "persona_scores" in df.columns
                            else [{} for _ in range(len(df))])
    df["triggered_by"] = df["triggered_by"].apply(lambda x: list(x) if x is not None else [])
    return df


@ttl_cache(300, maxsize=16)
def flags_lean(run: str, rnd: int) -> pd.DataFrame:
    """Every window of flags.parquet, identifying columns + tier only (for counts)."""
    p = round_dir(run, rnd) / "flags.parquet"
    if not p.exists():
        return pd.DataFrame()
    import pyarrow.parquet as pq
    cols = [c for c in FLAG_LEAN if c in pq.ParquetFile(p).schema_arrow.names]
    return pd.read_parquet(p, columns=cols)


def _in_market(f: pd.DataFrame, run: str, rnd: int, market: str | None) -> pd.DataFrame:
    """flags.parquet covers all test markets of a round; keep the selected market's tick range."""
    if f.empty or market is None:
        return f
    rng = tick_range(run, rnd, market)
    if rng is None:
        return f.iloc[0:0]
    return f[(f["window_start"] >= rng[0]) & (f["window_start"] <= rng[1])]


def flags_for_market(run: str, rnd: int, market: str | None) -> pd.DataFrame:
    return _in_market(flags(run, rnd), run, rnd, market)


def flags_lean_for_market(run: str, rnd: int, market: str | None) -> pd.DataFrame:
    return _in_market(flags_lean(run, rnd), run, rnd, market)


@ttl_cache(300, maxsize=32)
def thresholds(run: str, rnd: int) -> dict[str, float]:
    rd = round_dir(run, rnd)
    out: dict = {}
    for name in ("thresholds.json", "metrics.json"):
        p = rd / name
        if p.exists():
            data = json.loads(p.read_text(encoding="utf-8"))
            out = data.get("thresholds", {}) if name == "metrics.json" else data
            if out:
                break
    cfg = CONFIG_DIR / "detector_config.yaml"
    if cfg.exists():
        contamination = (yaml.safe_load(cfg.read_text()) or {}).get("isolation_forest", {}).get("contamination", 0.01)
        out = {**out, "anomaly": round(1.0 - float(contamination), 4)}
    return {k: float(v) for k, v in out.items() if isinstance(v, (int, float))}


def llm_config(section: str) -> dict:
    p = CONFIG_DIR / "llm_config.yaml"
    return (yaml.safe_load(p.read_text()) or {}).get(section, {}) if p.exists() else {}

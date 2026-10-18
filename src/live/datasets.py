"""
src/live/datasets.py
====================
Saved real-market datasets: the format the fetch script writes and the
prediction script reads.

    data/live/<dataset>/
        bars.parquet     commodity, ticker, ts_utc, open, high, low, close, volume
        manifest.json    source, interval, window, rows per commodity, notes

Dataset names carry the source, interval and the UTC window the bars cover,
e.g. yf_1m_20261016T1300Z_20261016T2100Z, so a fetch can be found and re-used later.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.live.bars import complete_grid, segment_real

REPO_ROOT = Path(__file__).resolve().parents[2]
LIVE_DIR = REPO_ROOT / "data" / "live"
BARS_FILE = "bars.parquet"
MANIFEST = "manifest.json"

# Front-month continuous futures on Yahoo Finance (same symbols as the calibration fetch)
TICKERS: dict[str, str] = {
    "crude_oil_wti": "CL=F",
    "copper": "HG=F",
    "natural_gas": "NG=F",
    "gold": "GC=F",
    "silver": "SI=F",
    "corn": "ZC=F",
    "brent_crude": "BZ=F",
    "wheat": "ZW=F",
}
SECTOR = {
    "crude_oil_wti": "energy", "brent_crude": "energy", "natural_gas": "energy",
    "gold": "metals", "silver": "metals", "copper": "metals",
    "corn": "grains", "wheat": "grains",
}


def dataset_name(source: str, interval: str, start: pd.Timestamp, end: pd.Timestamp) -> str:
    f = "%Y%m%dT%H%MZ"
    return f"{source}_{interval}_{start.tz_convert('UTC').strftime(f)}_{end.tz_convert('UTC').strftime(f)}"


def save_dataset(bars: pd.DataFrame, manifest: dict, name: str, root: Path = LIVE_DIR) -> Path:
    out = Path(root) / name
    out.mkdir(parents=True, exist_ok=True)
    bars.to_parquet(out / BARS_FILE, index=False)
    (out / MANIFEST).write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    return out


def list_datasets(root: Path = LIVE_DIR) -> list[Path]:
    return sorted((p for p in Path(root).glob("*") if (p / BARS_FILE).exists()),
                  key=lambda p: (p / MANIFEST).stat().st_mtime if (p / MANIFEST).exists() else 0)


def resolve_dataset(name: str | None, root: Path = LIVE_DIR) -> Path:
    """A dataset name, a path, or None/'latest' for the most recently fetched one."""
    if name and name != "latest":
        p = Path(name)
        p = p if (p / BARS_FILE).exists() else Path(root) / name
        if not (p / BARS_FILE).exists():
            raise FileNotFoundError(f"Dataset not found: {name}")
        return p
    found = list_datasets(root)
    if not found:
        raise FileNotFoundError("No saved datasets. Fetch one first: python scripts/fetch_market_data.py --hours 6")
    return found[-1]


def load_dataset(path: Path) -> tuple[pd.DataFrame, dict]:
    """
    Bars ready for feature computation: minute = minutes since the Unix epoch (UTC),
    contiguous segments numbered, missing minutes inside a segment filled flat with 0 volume.
    """
    path = Path(path)
    raw = pd.read_parquet(path / BARS_FILE)
    manifest = json.loads((path / MANIFEST).read_text(encoding="utf-8")) if (path / MANIFEST).exists() else {}
    ts = pd.to_datetime(raw["ts_utc"], utc=True)
    # Resolution-independent (parquet may round-trip timestamps as ns or us)
    minute = (ts - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta(minutes=1)
    bars = raw.assign(minute=minute.astype("int64"))
    bars = bars[["commodity", "minute", "open", "high", "low", "close", "volume"]]
    bars = complete_grid(segment_real(bars))
    return bars, manifest

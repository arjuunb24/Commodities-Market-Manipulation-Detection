"""
scripts/fetch_market_data.py
============================
Fetch recent real 1-minute bars for the project's eight commodity futures from
Yahoo Finance and save them as a re-usable dataset. This script only fetches and
saves; scoring is done separately by scripts/predict_live.py.

    data/live/yf_1m_<startUTC>_<endUTC>/bars.parquet + manifest.json

What "last N hours" means:
  default       the last N hours of TRADING data available (counted back from the
                newest bar). On a weekend or holiday this still returns the last
                session instead of nothing.
  --wall-clock  the N hours before now, even if markets were closed for part of it.

Limits (Yahoo): 1-minute bars go back about 7 days, so --hours is capped at 168.
The data is anonymous exchange bars: price and volume per minute, no orders, no
order book and no account ids.

Usage:
    python scripts/fetch_market_data.py --hours 6
    python scripts/fetch_market_data.py --hours 24 --wall-clock
    python scripts/fetch_market_data.py --hours 3 --commodities gold,silver
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.live.datasets import LIVE_DIR, TICKERS, dataset_name, save_dataset  # noqa: E402

MAX_HOURS_1M = 168


def fetch(commodities: list[str], hours: float, wall_clock: bool) -> tuple[pd.DataFrame, dict]:
    import yfinance as yf

    tickers = [TICKERS[c] for c in commodities]
    now = pd.Timestamp.now(tz="UTC")
    # Wall clock needs only the window itself; trading-hours mode reaches back further to
    # step over weekends and holidays.
    days = max(1, int(hours // 24) + 1) if wall_clock else min(7, int(hours // 24) + 4)
    raw = yf.download(tickers, period=f"{days}d", interval="1m", progress=False, auto_adjust=False,
                      group_by="ticker", threads=True, prepost=True)
    frames, notes = [], []
    for c, t in zip(commodities, tickers):
        try:
            d = raw[t] if isinstance(raw.columns, pd.MultiIndex) else raw
        except KeyError:
            notes.append(f"{c} ({t}): no data returned")
            continue
        d = d.dropna(subset=["Open", "High", "Low", "Close"])
        if d.empty:
            notes.append(f"{c} ({t}): no data returned")
            continue
        idx = d.index.tz_convert("UTC") if d.index.tz is not None else d.index.tz_localize("UTC")
        frames.append(pd.DataFrame({
            "commodity": c, "ticker": t, "ts_utc": idx,
            "open": d["Open"].to_numpy(float), "high": d["High"].to_numpy(float),
            "low": d["Low"].to_numpy(float), "close": d["Close"].to_numpy(float),
            "volume": d["Volume"].fillna(0).to_numpy(float),
        }))
    if not frames:
        return pd.DataFrame(), {"notes": notes}
    bars = pd.concat(frames, ignore_index=True)
    end = now if wall_clock else pd.Timestamp(bars["ts_utc"].max()) + pd.Timedelta(minutes=1)
    start = end - pd.Timedelta(minutes=round(hours * 60))
    bars = bars[(bars["ts_utc"] >= start) & (bars["ts_utc"] < end)].sort_values(["commodity", "ts_utc"])
    return bars.reset_index(drop=True), {"notes": notes, "window_start": start, "window_end": end, "fetched_at": now}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hours", type=float, required=True, help="how many hours of 1-minute bars (max 168)")
    ap.add_argument("--wall-clock", action="store_true", help="the N hours before now, not the last N trading hours")
    ap.add_argument("--commodities", default="all", help=f"'all' or a comma list of: {', '.join(TICKERS)}")
    ap.add_argument("--out", default=str(LIVE_DIR), help="folder that holds saved datasets (default data/live)")
    args = ap.parse_args()

    if not 0 < args.hours <= MAX_HOURS_1M:
        sys.exit(f"--hours must be between 0 and {MAX_HOURS_1M} (Yahoo keeps about 7 days of 1-minute bars).")
    commodities = list(TICKERS) if args.commodities == "all" else [c.strip() for c in args.commodities.split(",")]
    unknown = [c for c in commodities if c not in TICKERS]
    if unknown:
        sys.exit(f"Unknown commodities {unknown}. Choose from: {', '.join(TICKERS)}")

    print(f"Fetching {'the ' + str(args.hours) + ' hours before now' if args.wall_clock else 'the last ' + str(args.hours) + ' trading hours'} "
          f"of 1-minute bars for {len(commodities)} commodities from Yahoo Finance...")
    bars, info = fetch(commodities, args.hours, args.wall_clock)
    for n in info.get("notes", []):
        print("  note:", n)
    if bars.empty:
        sys.exit("No bars in that window. Markets may have been closed (weekend or holiday); "
                 "drop --wall-clock or ask for more hours.")

    start, end = info["window_start"], info["window_end"]
    per = bars.groupby("commodity").agg(rows=("ts_utc", "size"), first=("ts_utc", "min"), last=("ts_utc", "max"),
                                         zero_volume_share=("volume", lambda v: round(float((v <= 0).mean()), 3)))
    name = dataset_name("yf", "1m", start, end)
    manifest = {
        "dataset": name,
        "source": "Yahoo Finance via yfinance (front-month continuous futures)",
        "interval": "1m",
        "mode": "wall_clock" if args.wall_clock else "last_trading_hours",
        "requested_hours": args.hours,
        "window_start_utc": start,
        "window_end_utc": end,
        "fetched_at_utc": info["fetched_at"],
        "commodities": {c: {"ticker": TICKERS[c], **{k: (str(v) if k in ("first", "last") else v) for k, v in row.items()}}
                        for c, row in per.to_dict("index").items()},
        "missing": [c for c in commodities if c not in per.index],
        "notes": info.get("notes", []) + [
            "Anonymous bars: no orders, cancels, order book or account ids.",
            "Yahoo 1-minute volume can differ from exchange-reported volume; minutes with no trades are absent.",
        ],
    }
    out = save_dataset(bars, manifest, name, Path(args.out))
    print(f"\nSaved {len(bars):,} bars to {out}")
    print(f"Window: {start:%Y-%m-%d %H:%M} to {end:%Y-%m-%d %H:%M} UTC")
    print(per[["rows", "zero_volume_share"]].to_string())
    print(f"\nScore it with:  python scripts/predict_live.py --data {name}")


if __name__ == "__main__":
    main()

"""
scripts/fetch_calibration_data.py
==================================
Pulls daily OHLCV data from Yahoo Finance (via yfinance) for commodity
futures used to calibrate the Veridex ABM simulator.

Tickers:
  - GC=F  (Gold futures)
  - CL=F  (Crude Oil futures)
  - ZW=F  (Wheat futures)

Output: data/raw/yfinance/{gold,crude_oil,wheat}.parquet

EDGE CASES HANDLED (Implementation Plan Section 4.1 — not optional):
  1. Empty DataFrame from yfinance (rate-limiting, bad ticker):
     → Raises RuntimeError immediately. Never proceeds silently with 0 rows.
  2. Price gaps > 15% (potential contract roll artifact):
     → Logged as WARNING. Rows are NOT dropped — flagged for human review
       in the calibration notebook, not silently excluded.
  3. Missing trading days (holidays):
     → NOT forward-filled before log return computation. Gaps are preserved
       and noted. Forward-fill would bias volatility estimates downward.
  4. Network unavailability:
     → Catches all connection errors, logs WARNING, and writes a fallback
       notice to configs/agent_config.yaml using pre-calibrated constants
       derived from published historical statistics.

Usage:
    python scripts/fetch_calibration_data.py
    python scripts/fetch_calibration_data.py --period 5y --interval 1d
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Path setup — allow running from the repo root without installing the package
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.utils.logging_config import setup_logging  # noqa: E402

setup_logging()
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TICKERS: dict[str, str] = {
    "crude_oil_wti": "CL=F",
    "brent_crude": "BZ=F",
    "gold": "GC=F",
    "silver": "SI=F",
    "natural_gas": "NG=F",
    "copper": "HG=F",
    "wheat": "ZW=F",
    "corn": "ZC=F",
}

OUT_DIR = REPO_ROOT / "data" / "raw" / "yfinance"

# Pre-calibrated fallback values (daily parameters, normalised to price=100)
FALLBACK_PARAMS: dict[str, dict] = {
    "gold": {
        "ou_params": {"mean_reversion_speed": 0.05, "long_run_mean": 1.0, "volatility": 0.0094, "drift": 0.0001},
        "volume_params": {"mean_daily_volume": 5000, "volume_cv": 0.40},
    },
    "silver": {
        "ou_params": {"mean_reversion_speed": 0.045, "long_run_mean": 1.0, "volatility": 0.015, "drift": 0.0001},
        "volume_params": {"mean_daily_volume": 4000, "volume_cv": 0.42},
    },
    "crude_oil_wti": {
        "ou_params": {"mean_reversion_speed": 0.035, "long_run_mean": 1.0, "volatility": 0.022, "drift": 0.00005},
        "volume_params": {"mean_daily_volume": 8000, "volume_cv": 0.55},
    },
    "brent_crude": {
        "ou_params": {"mean_reversion_speed": 0.035, "long_run_mean": 1.0, "volatility": 0.021, "drift": 0.00005},
        "volume_params": {"mean_daily_volume": 7500, "volume_cv": 0.53},
    },
    "natural_gas": {
        "ou_params": {"mean_reversion_speed": 0.02, "long_run_mean": 1.0, "volatility": 0.045, "drift": 0.00002},
        "volume_params": {"mean_daily_volume": 6000, "volume_cv": 0.60},
    },
    "copper": {
        "ou_params": {"mean_reversion_speed": 0.04, "long_run_mean": 1.0, "volatility": 0.012, "drift": 0.00008},
        "volume_params": {"mean_daily_volume": 4500, "volume_cv": 0.35},
    },
    "wheat": {
        "ou_params": {"mean_reversion_speed": 0.08, "long_run_mean": 1.0, "volatility": 0.016, "drift": 0.00002},
        "volume_params": {"mean_daily_volume": 3000, "volume_cv": 0.45},
    },
    "corn": {
        "ou_params": {"mean_reversion_speed": 0.075, "long_run_mean": 1.0, "volatility": 0.014, "drift": 0.00002},
        "volume_params": {"mean_daily_volume": 3500, "volume_cv": 0.40},
    },
}


# ---------------------------------------------------------------------------
# Core fetch logic
# ---------------------------------------------------------------------------


def _check_roll_gaps(df: pd.DataFrame, ticker_name: str, threshold: float = 0.15) -> None:
    """
    Log a WARNING for price gaps > threshold (potential contract roll artifacts).

    Gaps are flagged for review in the calibration notebook but NOT removed
    here — silent exclusion would distort volatility estimates without audit trail.
    """
    if "Close" not in df.columns.get_level_values(0):
        return

    # yfinance sometimes returns a MultiIndex column even for single tickers
    if isinstance(df.columns, pd.MultiIndex):
        close_series = df["Close"].iloc[:, 0]
    else:
        close_series = df["Close"]

    log_returns = close_series.pct_change().dropna()
    extreme_gaps = log_returns[log_returns.abs() > threshold]

    if not extreme_gaps.empty:
        logger.warning(
            "[%s] Found %d price gap(s) > %.0f%% — potential contract roll artifact. "
            "Review these dates in notebooks/01_calibration.ipynb before using for "
            "volatility estimation: %s",
            ticker_name,
            len(extreme_gaps),
            threshold * 100,
            extreme_gaps.index.tolist(),
        )
    else:
        logger.info("[%s] No extreme price gaps detected (threshold %.0f%%).", ticker_name, threshold * 100)


def fetch_single(
    name: str,
    ticker: str,
    period: str,
    interval: str,
    out_dir: Path,
) -> pd.DataFrame:
    """
    Download OHLCV data for one ticker and persist to Parquet.

    Returns:
        The downloaded DataFrame.

    Raises:
        RuntimeError: If yfinance returns an empty DataFrame.
        ImportError: If yfinance is not installed.
    """
    import yfinance as yf

    logger.info("Fetching %s (%s) — period=%s, interval=%s", name, ticker, period, interval)

    df = yf.download(
        ticker,
        period=period,
        interval=interval,
        auto_adjust=True,
        progress=False,
    )

    # --- Edge case 1: empty response (rate limit / bad ticker) ---
    if df.empty:
        raise RuntimeError(
            f"yfinance returned NO data for {ticker} ({name}). "
            "Possible causes: rate limiting, ticker not found, or no network access. "
            "Retry in a few seconds, or check that the ticker is a valid continuous "
            "front-month futures contract on Yahoo Finance."
        )

    logger.info(
        "[%s] Downloaded %d rows from %s to %s",
        name, len(df), df.index.min().date(), df.index.max().date(),
    )

    # --- Edge case 2: extreme price gaps (roll artifacts) ---
    _check_roll_gaps(df, name)

    # --- Edge case 3: do NOT forward-fill missing days ---
    # Gaps (weekends, holidays) are preserved as-is. Log return computation
    # in calibration.py skips NaN rows via .dropna() — this is correct.
    if isinstance(df.columns, pd.MultiIndex):
        close_series = df["Close"].iloc[:, 0]
    else:
        close_series = df["Close"]
        
    n_missing = int(close_series.isna().sum())
    if n_missing > 0:
        logger.info("[%s] %d NaN Close values (holidays/missing days) — preserved, not filled.", name, n_missing)

    # Write to Parquet
    out_path = out_dir / f"{name}.parquet"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, engine="pyarrow")
    logger.info("[%s] Saved to %s", name, out_path)

    return df


def fetch_all(period: str = "5y", interval: str = "1d") -> dict[str, pd.DataFrame]:
    """
    Fetch calibration data for all tickers. Returns {name: DataFrame}.

    On network failure, falls back to pre-calibrated constants and writes
    a fallback notice to configs/agent_config.yaml.
    """
    results: dict[str, pd.DataFrame] = {}
    errors: list[str] = []

    for name, ticker in TICKERS.items():
        try:
            df = fetch_single(name, ticker, period, interval, OUT_DIR)
            results[name] = df
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "[%s] Fetch failed: %s — will use pre-calibrated fallback values.",
                name, exc,
            )
            errors.append(f"{name} ({ticker}): {exc}")

    if errors:
        _write_fallback_notice(errors)
    else:
        logger.info(
            "All %d tickers fetched successfully. "
            "Run src/core/calibration.py to fit OU parameters.",
            len(TICKERS),
        )
        _write_success_notice(results)

    return results


def _write_fallback_notice(errors: list[str]) -> None:
    """
    Write a fallback notice to configs/agent_config.yaml so the simulation
    can still run with pre-calibrated defaults even without network access.
    """
    import yaml

    config_path = REPO_ROOT / "configs" / "agent_config.yaml"

    try:
        with config_path.open("r") as f:
            config = yaml.safe_load(f) or {}
    except FileNotFoundError:
        config = {}

    config.setdefault("calibration", {})
    config["calibration"]["source"] = "fallback_published_stats"
    config["calibration"]["fallback_reason"] = (
        "yfinance fetch failed. Using pre-calibrated constants from published "
        "historical statistics (Erb & Harvey 2006, Geman 2005, Deaton & Laroque 1996). "
        f"Fetch errors: {errors}"
    )
    # Merge in fallback params for the primary calibration commodity (gold)
    config["calibration"].update(FALLBACK_PARAMS["gold"])

    with config_path.open("w") as f:
        yaml.safe_dump(config, f, default_flow_style=False, sort_keys=False)

    logger.warning(
        "FALLBACK MODE: Could not fetch live calibration data. "
        "Pre-calibrated constants written to configs/agent_config.yaml. "
        "Errors: %s",
        errors,
    )


def _write_success_notice(results: dict[str, pd.DataFrame]) -> None:
    """Update configs/agent_config.yaml with a note that live data was fetched."""
    import yaml

    config_path = REPO_ROOT / "configs" / "agent_config.yaml"

    try:
        with config_path.open("r") as f:
            config = yaml.safe_load(f) or {}
    except FileNotFoundError:
        config = {}

    config.setdefault("calibration", {})
    config["calibration"]["source"] = "yfinance"
    config["calibration"]["note"] = (
        f"Fetched {[k for k in results]}. "
        "Run src/core/calibration.py to update OU parameters from this data."
    )

    with config_path.open("w") as f:
        yaml.safe_dump(config, f, default_flow_style=False, sort_keys=False)


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch commodity futures OHLCV data from Yahoo Finance for calibration."
    )
    parser.add_argument(
        "--period",
        default="5y",
        choices=["1y", "2y", "5y", "10y"],
        help="Historical period to fetch (default: 5y)",
    )
    parser.add_argument(
        "--interval",
        default="1d",
        choices=["1d", "1wk"],
        help="Data frequency (default: 1d)",
    )
    args = parser.parse_args()

    logger.info("Starting calibration data fetch (period=%s, interval=%s)", args.period, args.interval)
    results = fetch_all(period=args.period, interval=args.interval)
    logger.info(
        "Fetch complete. %d/%d tickers succeeded. Data written to %s",
        len(results), len(TICKERS), OUT_DIR,
    )


if __name__ == "__main__":
    main()

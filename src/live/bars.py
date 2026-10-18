"""
src/live/bars.py
================
1-minute OHLCV bars and the scale-free features computed from them.

The same feature code runs on simulator bars (training) and on real exchange
bars (prediction), so every feature is relative to the series' own recent
behaviour: returns in units of local volatility, volume against its own median,
range against its own median range. Absolute prices and volumes never enter.

Surveillance here is retrospective (reviewing a finished window, like a T+1
review), so features look both back and forward from each minute: a pump and
dump is only recognisable once the dump has happened.

Bars table (long format), shared by simulator and real data:
    commodity, minute (int, minutes since an arbitrary origin), open, high, low,
    close, volume, segment (int; bars in different segments are not contiguous,
    e.g. across the daily exchange halt). Real data uses minutes since the Unix
    epoch (UTC), so `minute_to_utc` turns a minute back into a timestamp.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

BAR_SECONDS = 60                 # simulator: 1 tick = 1 second
BASELINE_MIN = 60                # trailing minutes for "normal" volatility / volume / range
MIN_BASELINE = 20                # bars needed before a baseline is trusted
LABEL_MIN_OVERLAP_S = 30         # a minute is labelled with an episode it overlaps by >= 30 s
GAP_SPLIT_MIN = 10               # real data: a gap longer than this starts a new segment
SIGMA_FLOOR_BP = 0.5             # volatility floor so flat stretches do not explode z-scores

FEATURES_V1 = [
    "ret_z_1", "ret_z_back_5", "ret_z_back_15", "ret_z_fwd_5", "ret_z_fwd_15",
    "range_ratio", "range_ratio_5", "vol_z", "vol_ratio_5", "vol_ratio_15",
    "body_share", "wick_share_5", "vol_per_move", "drawup_z_15", "drawdown_z_15",
    "reversal_5", "reversal_15", "up_share_10", "pv_corr_15", "zero_vol_share_15",
    "xs_ret_z_5", "xs_comove_share",
]
# v2 (added, v1 unchanged so earlier saved models still score): manipulators are only 3-13%
# of an episode's volume, so a single minute shows little; these aggregate over episode-length
# windows and borrow microstructure estimators that work on bars.
#   multi-scale returns, centred volume / |return| / range windows
#   bulk volume classification order-flow imbalance (signed flow from bar returns)
#   price impact: Kyle-lambda-style slope of return on signed flow, Amihud |return| per volume
#   spreads: Roll (negative return autocovariance), Corwin-Schultz (high-low)
#   variance ratio (mean reversion) and bipower jump share
#   volume unexplained by volatility (log volume residual on |return|): wash trading's footprint
FEATURES_V2_EXTRA = [
    "ret_z_back_3", "ret_z_fwd_3", "ret_z_back_10", "ret_z_fwd_10", "ret_z_back_30", "ret_z_fwd_30",
    "vrel_c5", "absz_c5", "vrel_c9", "absz_c9", "vrel_c15", "absz_c15", "range_c9",
    "ofi_5", "ofi_c5", "ofi_15", "ofi_c15", "kyle_15", "amihud_c5", "amihud_ratio",
    "roll_ac_15", "cs_spread_c5", "var_ratio_15", "jump_share_15",
    "vol_resid", "vol_resid_c5", "vol_resid_c9", "pos_30",
]
FEATURES = FEATURES_V1 + FEATURES_V2_EXTRA
BARS_CACHE = "bars_1m.parquet"


# ---------------------------------------------------------------------------
# Bars
# ---------------------------------------------------------------------------

def sim_bars(trade_df: pd.DataFrame) -> pd.DataFrame:
    """1-minute bars per commodity from a simulator trade log (tick = 1 second)."""
    if trade_df.empty:
        return pd.DataFrame(columns=["commodity", "minute", "open", "high", "low", "close", "volume", "segment"])
    t = trade_df[["commodity", "tick", "price", "quantity"]].sort_values(["commodity", "tick"])
    t = t.assign(minute=(t["tick"] // BAR_SECONDS).astype("int64"))
    g = t.groupby(["commodity", "minute"])
    bars = pd.DataFrame({
        "open": g["price"].first(), "high": g["price"].max(), "low": g["price"].min(),
        "close": g["price"].last(), "volume": g["quantity"].sum(),
    }).reset_index()
    return complete_grid(bars)


def complete_grid(bars: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing minutes inside each commodity's segments: price carried forward as a
    flat bar, volume 0 (both exchanges and the simulator simply print nothing then).
    """
    if "segment" not in bars:
        bars = bars.assign(segment=0)
    out = []
    for (c, seg), b in bars.groupby(["commodity", "segment"], sort=False):
        b = b.set_index("minute").sort_index()
        idx = pd.RangeIndex(int(b.index.min()), int(b.index.max()) + 1, name="minute")
        b = b.reindex(idx)
        b["close"] = b["close"].ffill()
        for col in ("open", "high", "low"):
            b[col] = b[col].fillna(b["close"])
        b["volume"] = b["volume"].fillna(0.0)
        b["commodity"] = c
        b["segment"] = seg
        out.append(b.reset_index())
    cols = ["commodity", "minute", "open", "high", "low", "close", "volume", "segment"]
    res = pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=cols)
    return res[cols]


def minute_to_utc(minute) -> pd.Timestamp:
    """Real-data minute index (minutes since the Unix epoch) → UTC timestamp."""
    return pd.to_datetime(np.asarray(minute, dtype="int64") * 60, unit="s", utc=True)


def segment_real(bars: pd.DataFrame) -> pd.DataFrame:
    """Number contiguous stretches per commodity: a gap over GAP_SPLIT_MIN minutes starts a new one."""
    bars = bars.sort_values(["commodity", "minute"]).reset_index(drop=True)
    gap = bars.groupby("commodity")["minute"].diff().fillna(GAP_SPLIT_MIN + 1)
    bars["segment"] = (gap > GAP_SPLIT_MIN).groupby(bars["commodity"]).cumsum().astype(int)
    return bars


def bars_for_market(market_dir: Path, use_cache: bool = True) -> pd.DataFrame:
    """Bars of one simulated market directory, cached next to its logs."""
    market_dir = Path(market_dir)
    cache, src = market_dir / BARS_CACHE, market_dir / "trade_log.parquet"
    if use_cache and cache.exists() and src.exists() and cache.stat().st_mtime >= src.stat().st_mtime:
        return pd.read_parquet(cache)
    bars = sim_bars(pd.read_parquet(src))
    bars.to_parquet(cache, index=False)
    return bars


def episode_labels(bars: pd.DataFrame, events: pd.DataFrame, personas: list[str]) -> pd.DataFrame:
    """One 0/1 column per persona: the minute overlaps an episode of that persona by >= 30 s."""
    y = pd.DataFrame(0, index=bars.index, columns=personas, dtype=int)
    if events is None or events.empty or bars.empty:
        return y
    start = bars["minute"].to_numpy() * BAR_SECONDS
    end = start + BAR_SECONDS
    for _, ev in events.iterrows():
        if ev["persona"] not in personas:
            continue
        m = (bars["commodity"].to_numpy() == ev["commodity"])
        overlap = np.minimum(end, ev["end_tick"]) - np.maximum(start, ev["start_tick"])
        hit = m & (overlap >= LABEL_MIN_OVERLAP_S)
        y.loc[hit, ev["persona"]] = 1
    return y


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------

def _roll(s: pd.Series, n: int, fn: str, min_periods: int | None = None) -> pd.Series:
    return getattr(s.rolling(n, min_periods=min_periods or max(1, n // 2)), fn)()


def _centered(s: pd.Series, n: int, fn: str = "mean") -> pd.Series:
    return getattr(s.rolling(n, center=True, min_periods=1), fn)()


def _fwd_sum(s: pd.Series, n: int) -> pd.Series:
    """Sum of the next n values (excluding the current one)."""
    return s[::-1].rolling(n, min_periods=1).sum()[::-1].shift(-1)


def _series_features(b: pd.DataFrame) -> pd.DataFrame:
    """Features of one contiguous commodity segment (rows sorted by minute)."""
    close = b["close"].astype(float)
    ret = np.log(close).diff().fillna(0.0) * 1e4                      # bp
    sigma = _roll(ret, BASELINE_MIN, "std", MIN_BASELINE).shift(1)
    sigma = sigma.fillna(ret.std() if len(ret) > 1 else SIGMA_FLOOR_BP).clip(lower=SIGMA_FLOOR_BP)
    rng = ((b["high"] - b["low"]) / close * 1e4).astype(float)
    rng_med = _roll(rng, BASELINE_MIN, "median", MIN_BASELINE).shift(1).fillna(rng.median()).clip(lower=0.1)
    vol = b["volume"].astype(float)
    lv = np.log1p(vol)
    lv_med = _roll(lv, BASELINE_MIN, "median", MIN_BASELINE).shift(1).fillna(lv.median())
    lv_mad = (lv - lv_med).abs()
    lv_scale = (_roll(lv_mad, BASELINE_MIN, "median", MIN_BASELINE).shift(1).fillna(lv_mad.median()) * 1.4826).clip(lower=0.25)
    vol_med = np.expm1(lv_med).clip(lower=1.0)

    f = pd.DataFrame(index=b.index)
    f["ret_z_1"] = ret / sigma
    for k in (5, 15):
        f[f"ret_z_back_{k}"] = _roll(ret, k, "sum", 1) / (sigma * np.sqrt(k))
        f[f"ret_z_fwd_{k}"] = _fwd_sum(ret, k).fillna(0.0) / (sigma * np.sqrt(k))
    f["range_ratio"] = rng / rng_med
    f["range_ratio_5"] = _roll(rng, 5, "mean", 1) / rng_med
    f["vol_z"] = (lv - lv_med) / lv_scale
    f["vol_ratio_5"] = _roll(vol, 5, "mean", 1) / vol_med
    f["vol_ratio_15"] = _roll(vol, 15, "mean", 1) / vol_med
    span = (b["high"] - b["low"]).astype(float)
    body = (b["close"] - b["open"]).abs().astype(float)
    f["body_share"] = np.where(span > 0, body / span.where(span > 0, 1.0), 1.0)
    f["wick_share_5"] = 1.0 - _roll(f["body_share"], 5, "mean", 1)
    f["vol_per_move"] = f["vol_z"] - f["ret_z_1"].abs()
    lo15, hi15 = _roll(close, 15, "min", 1), _roll(close, 15, "max", 1)
    f["drawup_z_15"] = np.log(close / lo15) * 1e4 / (sigma * np.sqrt(15))
    f["drawdown_z_15"] = np.log(hi15 / close) * 1e4 / (sigma * np.sqrt(15))
    # Reversal: a move into this minute that the following minutes give back (positive = reversal)
    for k in (5, 15):
        f[f"reversal_{k}"] = -np.sign(f[f"ret_z_back_{k}"]) * f[f"ret_z_fwd_{k}"]
    f["up_share_10"] = _roll((ret > 0).astype(float), 10, "mean", 1)
    pv = ret.abs().rolling(15, min_periods=5).corr(vol)
    f["pv_corr_15"] = pv.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    f["zero_vol_share_15"] = _roll((vol <= 0).astype(float), 15, "mean", 1)
    _v2_features(f, b, ret, sigma, vol)
    return f.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _v2_features(f: pd.DataFrame, b: pd.DataFrame, ret: pd.Series, sigma: pd.Series, vol: pd.Series) -> None:
    """Episode-length aggregates and bar-based microstructure estimators (see FEATURES_V2_EXTRA)."""
    from scipy.stats import norm

    close = b["close"].astype(float)
    vmed = _roll(vol, BASELINE_MIN, "median", MIN_BASELINE).shift(1).fillna(vol.median()).clip(lower=1.0)
    vrel = vol / vmed
    z = ret / sigma
    for k in (3, 10, 30):
        f[f"ret_z_back_{k}"] = _roll(ret, k, "sum", 1) / (sigma * np.sqrt(k))
        f[f"ret_z_fwd_{k}"] = _fwd_sum(ret, k).fillna(0.0) / (sigma * np.sqrt(k))
    for k in (5, 9, 15):
        f[f"vrel_c{k}"] = _centered(vrel, k)
        f[f"absz_c{k}"] = _centered(z.abs(), k)
    rng = (b["high"] - b["low"]) / close * 1e4
    rmed = _roll(rng, BASELINE_MIN, "median", MIN_BASELINE).shift(1).fillna(rng.median()).clip(lower=0.1)
    f["range_c9"] = _centered(rng / rmed, 9)
    # Bulk volume classification: the buy share of a bar's volume is Phi(return / sigma)
    signed = vol * (2 * norm.cdf(z.clip(-6, 6)) - 1)
    for k in (5, 15):
        f[f"ofi_{k}"] = _roll(signed, k, "sum", 1) / _roll(vol, k, "sum", 1).clip(lower=1.0)
        f[f"ofi_c{k}"] = _centered(signed, k, "sum") / _centered(vol, k, "sum").clip(lower=1.0)
    sv = signed / vmed
    w = lambda s: s.rolling(15, min_periods=5)  # noqa: E731
    cov = w(z * sv).mean() - w(z).mean() * w(sv).mean()
    f["kyle_15"] = (cov / w(sv).var().replace(0, np.nan)).fillna(0.0).clip(-20, 20)
    amihud = z.abs() / vrel.clip(lower=0.05)
    f["amihud_c5"] = _centered(amihud, 5)
    a_med = _roll(amihud, BASELINE_MIN, "median", MIN_BASELINE).shift(1).fillna(amihud.median()).clip(lower=0.05)
    f["amihud_ratio"] = _centered(amihud, 9) / a_med
    f["roll_ac_15"] = (-w(ret * ret.shift(1)).mean() / sigma ** 2).clip(-5, 5).fillna(0.0)
    hl = np.log(b["high"] / b["low"]).astype(float)
    beta = hl ** 2 + hl.shift(1) ** 2
    h2 = np.log(pd.concat([b["high"], b["high"].shift(1)], axis=1).max(axis=1)
                / pd.concat([b["low"], b["low"].shift(1)], axis=1).min(axis=1)).astype(float)
    k2 = 3 - 2 * np.sqrt(2)
    alpha = (np.sqrt(2 * beta) - np.sqrt(beta)) / k2 - np.sqrt(h2 ** 2 / k2)
    cs = (2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))).clip(lower=0).fillna(0.0) * 1e4
    cs_med = _roll(cs, BASELINE_MIN, "median", MIN_BASELINE).shift(1).fillna(cs.median()).clip(lower=0.1)
    f["cs_spread_c5"] = _centered(cs, 5) / cs_med
    rv1 = w(ret ** 2).sum()
    rv5 = w(_roll(ret, 5, "sum", 1) ** 2).mean() * 3
    f["var_ratio_15"] = (rv5 / rv1.clip(lower=1e-6)).clip(0, 5).fillna(1.0)
    bv = (np.pi / 2) * w(ret.abs() * ret.abs().shift(1)).sum()
    f["jump_share_15"] = ((rv1 - bv) / rv1.clip(lower=1e-6)).clip(-1, 1).fillna(0.0)
    # Volume the price move does not explain: residual of log volume on |z| (trailing fit)
    lv, x = np.log1p(vol), z.abs()
    roll = lambda s, fn: getattr(s.rolling(BASELINE_MIN, min_periods=MIN_BASELINE), fn)().shift(1)  # noqa: E731
    mx, my = roll(x, "mean"), roll(lv, "mean")
    slope = ((roll(x * lv, "mean") - mx * my) / roll(x, "var").replace(0, np.nan)).fillna(0.0)
    f["vol_resid"] = (lv - (my + slope * (x - mx))).fillna(0.0)
    f["vol_resid_c5"] = _centered(f["vol_resid"], 5)
    f["vol_resid_c9"] = _centered(f["vol_resid"], 9)
    hi30, lo30 = _roll(close, 30, "max", 1), _roll(close, 30, "min", 1)
    f["pos_30"] = ((close - lo30) / (hi30 - lo30).replace(0, np.nan)).fillna(0.5)


def bar_features(bars: pd.DataFrame) -> pd.DataFrame:
    """
    Feature matrix aligned to `bars` rows. Cross-commodity features compare each
    commodity with the others at the same minute: real moves driven by news or
    macro data usually hit a whole sector at once; manipulation of one book does not.
    """
    bars = bars.reset_index(drop=True)
    parts = []
    for _, b in bars.groupby(["commodity", "segment"], sort=False):
        parts.append(_series_features(b.sort_values("minute")))
    f = pd.concat(parts).reindex(bars.index) if parts else pd.DataFrame(index=bars.index)
    key = bars["minute"]
    r5 = f["ret_z_back_5"]
    med = r5.groupby(key).transform("median")
    n = r5.groupby(key).transform("count")
    f["xs_ret_z_5"] = np.where(n > 2, r5 - med, r5)
    big_up = (r5 >= 2).astype(int).groupby(key).transform("sum")
    big_dn = (r5 <= -2).astype(int).groupby(key).transform("sum")
    same = np.where(r5 >= 2, big_up - 1, np.where(r5 <= -2, big_dn - 1, 0))
    f["xs_comove_share"] = np.where(n > 1, same / (n - 1).clip(lower=1), 0.0)
    return f[FEATURES].astype(float)

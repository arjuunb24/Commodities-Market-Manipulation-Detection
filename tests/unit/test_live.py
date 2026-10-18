"""
tests/unit/test_live.py
=======================
Real-market testing features: bars and bar features, the bar-level model and its
alert budget, the model registry, the mechanism checker and market calendar,
case building, saved datasets, and the fetch script (Yahoo mocked).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.live import bars as B
from src.live.bar_model import ALERT_BUDGET, BarModel, _best_threshold
from src.live.calendar import events_overlapping
from src.live.checker import MechanismChecker
from src.live.datasets import load_dataset, save_dataset
from src.live.predict import build_cases
from src.live.registry import load_bar_model, load_manifest, save_round_models

REPO_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _series(n=180, seed=0, commodity="gold", start_minute=0, price=100.0, vol=100.0, sigma_bp=2.0):
    rng = np.random.default_rng(seed)
    close = price * np.exp(np.cumsum(rng.normal(0, sigma_bp / 1e4, n)))
    open_ = np.r_[price, close[:-1]]
    hi = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, sigma_bp / 2e4, n)))
    lo = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, sigma_bp / 2e4, n)))
    v = rng.poisson(vol, n).astype(float)
    return pd.DataFrame({"commodity": commodity, "minute": np.arange(start_minute, start_minute + n),
                         "open": open_, "high": hi, "low": lo, "close": close, "volume": v, "segment": 0})


# ---------------------------------------------------------------------------
# bars
# ---------------------------------------------------------------------------

def test_sim_bars_aggregate_and_fill_missing_minutes():
    trades = pd.DataFrame({
        "commodity": "gold", "tick": [0, 30, 59, 180, 200],
        "price": [10.0, 11.0, 9.0, 12.0, 12.5], "quantity": [1, 2, 3, 4, 5],
    })
    bars = B.sim_bars(trades)
    assert list(bars["minute"]) == [0, 1, 2, 3]
    first = bars.iloc[0]
    assert (first["open"], first["high"], first["low"], first["close"], first["volume"]) == (10.0, 11.0, 9.0, 9.0, 6)
    gap = bars[bars["minute"] == 1].iloc[0]
    assert gap["volume"] == 0 and gap["open"] == gap["close"] == 9.0      # flat bar, no trades


def test_episode_labels_need_30s_overlap():
    bars = pd.DataFrame({"commodity": "gold", "minute": [0, 1, 2, 3]})
    ev = pd.DataFrame([{"commodity": "gold", "persona": "spoofing", "start_tick": 80, "end_tick": 170}])
    y = B.episode_labels(bars, ev, ["spoofing", "layering"])
    # minute 1 = ticks 60-120 (40 s overlap), minute 2 = 120-180 (50 s), minute 0 and 3 none
    assert list(y["spoofing"]) == [0, 1, 1, 0]
    assert y["layering"].sum() == 0


def test_bar_features_are_scale_free_and_finite():
    a = _series(seed=1)
    b = a.copy()
    b[["open", "high", "low", "close"]] *= 37.0     # another price level
    b["volume"] *= 9.0                              # another volume level
    fa, fb = B.bar_features(a), B.bar_features(b)
    assert list(fa.columns) == B.FEATURES
    assert np.isfinite(fa.to_numpy()).all()
    # log1p(volume) features (vol_z, the volume residuals) keep tiny differences; everything else is identical
    log_vol = ("vol_z", "vol_per_move", "pv_corr_15", "vol_resid", "vol_resid_c5", "vol_resid_c9")
    cols = [c for c in B.FEATURES if c not in log_vol]
    np.testing.assert_allclose(fa[cols].to_numpy(), fb[cols].to_numpy(), atol=1e-6)


def test_cross_commodity_comove_feature():
    a, b = _series(seed=2, commodity="gold"), _series(seed=3, commodity="silver")
    for df in (a, b):           # both jump at minute 100: a market-wide move
        df.loc[df["minute"] >= 100, ["open", "high", "low", "close"]] *= 1.01
    f = B.bar_features(pd.concat([a, b], ignore_index=True))
    rows = pd.concat([a, b], ignore_index=True)["minute"] == 101
    assert (f.loc[rows, "xs_comove_share"] > 0).all()


def test_segment_real_splits_on_gaps():
    bars = pd.DataFrame({"commodity": "gold", "minute": [0, 1, 2, 60, 61], "open": 1.0, "high": 1.0,
                         "low": 1.0, "close": 1.0, "volume": 1.0})
    seg = B.segment_real(bars)
    assert list(seg["segment"]) == [1, 1, 1, 2, 2]


# ---------------------------------------------------------------------------
# bar model
# ---------------------------------------------------------------------------

def test_threshold_respects_alert_budget():
    rng = np.random.default_rng(0)
    y = (rng.random(5000) < 0.05).astype(int)
    p = np.clip(0.3 + 0.1 * y + rng.normal(0, 0.15, 5000), 0, 1)    # weak signal: F1-best would flag a lot
    thr = _best_threshold(y, p)
    assert (p >= thr).mean() <= ALERT_BUDGET + 0.002


def test_bar_model_learns_a_planted_signal():
    rng = np.random.default_rng(1)
    X = pd.DataFrame(rng.normal(size=(3000, len(B.FEATURES))), columns=B.FEATURES)
    y = pd.DataFrame({p: 0 for p in ["spoofing", "wash_trading", "pump_and_dump", "layering"]}, index=X.index)
    pos = X["vol_per_move"] > 1.6
    y.loc[pos, "wash_trading"] = 1
    groups = np.repeat([0, 1, 2], 1000)
    m = BarModel(["wash_trading"]).fit(X, y, groups)
    assert m.metrics["train_oof"]["wash_trading"]["auc"] > 0.9
    held = m.evaluate(X, y)
    assert held["wash_trading"]["precision"] > 0.5
    assert m.skill_summary()["level"] == "good"


def test_smoothing_stays_inside_one_series_and_old_models_skip_it():
    from src.live.bar_model import smooth_probs

    probs = pd.DataFrame({"spoofing": [0.0, 0.0, 1.0, 0.0, 0.0, 9.0]})
    keys = pd.DataFrame({"commodity": ["gold"] * 5 + ["silver"], "segment": 0})
    sm = smooth_probs(probs, keys, 3)["spoofing"].tolist()
    assert sm[1] == pytest.approx(1 / 3) and sm[3] == pytest.approx(1 / 3)
    assert sm[4] == 0.0 and sm[5] == 9.0                  # silver never mixes into gold
    old = BarModel(["spoofing"])
    del old.smooth_minutes                                # a model pickled before smoothing existed
    assert old._smooth == 1


def test_registry_round_trip_and_manifest_merge(tmp_path):
    rng = np.random.default_rng(2)
    X = pd.DataFrame(rng.normal(size=(600, len(B.FEATURES))), columns=B.FEATURES)
    y = pd.DataFrame({"spoofing": (X["ret_z_1"] > 1.2).astype(int)})
    m = BarModel(["spoofing"]).fit(X, y, np.repeat([0, 1], 300))
    rd = tmp_path / "adversarial_x" / "round_3"
    out = save_round_models(rd, bar_model=m, extra={"note": "first"})
    loaded = load_bar_model(out)
    np.testing.assert_allclose(loaded.predict_proba(X)["spoofing"], m.predict_proba(X)["spoofing"])
    man = load_manifest(out)
    assert man["round"] == 3 and man["run"] == "adversarial_x" and man["files"]["bar_model"] == "bar_model.joblib"
    save_round_models(rd, extra={"note": "second"})            # saving nothing new keeps the earlier entries
    man2 = load_manifest(out)
    assert man2["files"]["bar_model"] == "bar_model.joblib" and man2["note"] == "second"


# ---------------------------------------------------------------------------
# checker
# ---------------------------------------------------------------------------

def _with(df, minutes, *, mult=None, vol=None, wick=None):
    df = df.copy()
    m = df["minute"].isin(minutes)
    if mult is not None:
        for col in ("open", "high", "low", "close"):
            df.loc[m, col] = df.loc[m, col] * np.asarray(mult)
    if vol is not None:
        df.loc[m, "volume"] = vol
    return df


def test_checker_recognises_a_pump_and_dump():
    base = _series(n=200, seed=4, sigma_bp=1.0)
    up = np.linspace(1.0, 1.006, 10)          # +60 bp over 10 minutes (about 6 sigma)
    df = _with(base, range(100, 110), mult=up, vol=400)
    after = df["minute"] >= 110
    df.loc[after, ["open", "high", "low", "close"]] *= 1.001        # most of the run-up gone again
    res = MechanismChecker(df, real_time=False).check("gold", 100, 109)
    pd_res = res["personas"]["pump_and_dump"]
    assert pd_res["verdict"] == "consistent", pd_res["tests"]


def test_checker_wash_trading_is_at_best_unverifiable():
    df = _with(_series(n=200, seed=5, sigma_bp=2.0), range(120, 124), vol=1000)
    df.loc[df["minute"].between(120, 123), "high"] = df["close"] * 1.00002
    df.loc[df["minute"].between(120, 123), "low"] = df["close"] * 0.99998
    res = MechanismChecker(df, real_time=False).check("gold", 120, 123)
    assert res["personas"]["wash_trading"]["verdict"] == "consistent (unverifiable)"
    assert "account ids" in res["personas"]["wash_trading"]["data_needed_to_confirm"]


def test_checker_calls_a_sector_wide_move_benign():
    gold = _series(n=200, seed=6, commodity="gold", sigma_bp=1.0)
    silver = _series(n=200, seed=7, commodity="silver", sigma_bp=1.0)
    gold, silver = (_with(x, range(100, 110), mult=np.linspace(1.0, 1.006, 10), vol=400) for x in (gold, silver))
    for x in (gold, silver):
        x.loc[x["minute"] >= 110, ["open", "high", "low", "close"]] *= 1.001
    res = MechanismChecker(pd.concat([gold, silver], ignore_index=True), real_time=False).check("gold", 100, 109)
    assert res["personas"]["pump_and_dump"]["verdict"] == "likely benign"
    assert any("silver" in b for b in res["benign_explanations"])


def test_calendar_finds_the_eia_crude_report():
    # Wednesday 2026-10-14 10:30 ET = 14:30 UTC
    hits = events_overlapping("crude_oil_wti", pd.Timestamp("2026-10-14 14:32", tz="UTC"), pd.Timestamp("2026-10-14 14:36", tz="UTC"))
    assert any("EIA weekly petroleum" in h["event"] for h in hits)
    # not for gold, and not on a Tuesday
    assert not any("EIA" in h["event"] for h in events_overlapping("gold", pd.Timestamp("2026-10-14 14:32", tz="UTC"), pd.Timestamp("2026-10-14 14:36", tz="UTC")))
    assert not any("petroleum" in h["event"] for h in events_overlapping("crude_oil_wti", pd.Timestamp("2026-10-13 14:32", tz="UTC"), pd.Timestamp("2026-10-13 14:36", tz="UTC")))


# ---------------------------------------------------------------------------
# cases and datasets
# ---------------------------------------------------------------------------

def test_build_cases_merges_close_flags():
    s = pd.DataFrame({
        "commodity": "gold", "segment": 0, "minute": [1, 2, 4, 20, 21],
        "flag": True, "predicted_persona": ["spoofing", "spoofing", "layering", "wash_trading", "wash_trading"],
        "prob_spoofing": [0.9, 0.8, 0.3, 0.1, 0.1], "prob_layering": [0.1, 0.2, 0.7, 0.1, 0.1],
        "prob_wash_trading": [0.1, 0.1, 0.1, 0.9, 0.95], "margin": [0.2, 0.1, 0.05, 0.3, 0.35],
    })
    cases = build_cases(s, gap=2)
    assert [(c["start_minute"], c["end_minute"]) for c in cases] == [(20, 21), (1, 4)]
    assert cases[1]["model_persona"] == "spoofing" and cases[0]["model_persona"] == "wash_trading"


def test_saved_dataset_round_trip(tmp_path):
    ts = pd.date_range("2026-10-16 15:00", periods=5, freq="min", tz="UTC").delete(2)   # one missing minute
    raw = pd.DataFrame({"commodity": "gold", "ticker": "GC=F", "ts_utc": ts, "open": 1.0, "high": 1.1,
                        "low": 0.9, "close": 1.0, "volume": 10.0})
    out = save_dataset(raw, {"dataset": "t"}, "yf_1m_test", tmp_path)
    bars, man = load_dataset(out)
    assert len(bars) == 5 and bars["volume"].tolist()[2] == 0.0             # gap filled flat
    assert B.minute_to_utc(bars["minute"])[0] == pd.Timestamp("2026-10-16 15:00", tz="UTC")
    assert man["dataset"] == "t"


def test_fetch_keeps_the_last_trading_hours(monkeypatch):
    spec = importlib.util.spec_from_file_location("fetch_market_data", REPO_ROOT / "scripts" / "fetch_market_data.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    idx = pd.date_range("2026-10-16 10:00", periods=600, freq="min", tz="America/New_York")
    cols = pd.MultiIndex.from_product([["GC=F", "SI=F"], ["Open", "High", "Low", "Close", "Adj Close", "Volume"]])
    raw = pd.DataFrame(1.0, index=idx, columns=cols)

    class FakeYF:
        @staticmethod
        def download(*a, **k):
            return raw

    monkeypatch.setitem(sys.modules, "yfinance", FakeYF)
    bars, info = mod.fetch(["gold", "silver"], 2, wall_clock=False)
    assert set(bars["commodity"]) == {"gold", "silver"}
    assert bars.groupby("commodity").size().tolist() == [120, 120]          # exactly the last 2 hours
    assert info["window_end"] - info["window_start"] == pd.Timedelta(hours=2)

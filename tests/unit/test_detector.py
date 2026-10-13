"""
tests/unit/test_detector.py
===========================
Unit tests for the trader-level detector:
- FeatureEngine (hand-built logs → exact feature values, NaN/neutral handling)
- Labelling (trader-level ground truth)
- AnomalyDetector / SupervisedDetector / DetectorPipeline on a small simulated run
- Metrics (event-level detection, latency NaN for misses)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.adversarial.metrics import compute_round_metrics, event_detection
from src.core.simulation import SimConfig, run_simulation
from src.detector.anomaly import AnomalyDetector
from src.detector.classifiers import InsufficientLabelsError, SupervisedDetector
from src.detector.features import FeatureEngine
from src.detector.pipeline import DetectorPipeline, read_flags
from src.utils import io as io_utils

CFG = {
    "feature_engine": {"window": "60s", "step": "30s", "exclude_trader_prefixes": ["mm_"]},
    "isolation_forest": {"contamination": 0.05, "n_estimators": 50, "random_state": 0},
    "xgboost": {"n_estimators_small": 40, "max_depth_small": 3},
    "supervised": {"min_positives": 3, "cv_folds": 2, "cv_block_ticks": 300},
    "personas": ["spoofing", "wash_trading", "pump_and_dump"],
}
# Simulated-data tests use the production window (pump-and-dump episodes span minutes).
SIM_CFG = {**CFG, "feature_engine": {**CFG["feature_engine"], "window": "3min"}}


def _order(oid, trader, side, price, qty, tick, ev="placed"):
    return {"order_id": oid, "commodity": "gold", "trader_id": trader, "side": side,
            "price": price, "quantity": qty, "tick": tick, "event_type": ev}


def _trade(tid, tick, price, qty, buyer, seller, aggr="buy"):
    return {"trade_id": tid, "commodity": "gold", "tick": tick, "price": price, "quantity": qty,
            "buyer_trader_id": buyer, "seller_trader_id": seller, "aggressor_side": aggr}


@pytest.fixture
def tiny_logs():
    """Two 30-tick steps. 'spoofy' places a huge buy, gets filled selling, cancels the buy fast."""
    orders = [
        _order("s1", "spoofy", "buy", 99.0, 400, 2),
        _order("g1", "spoofy", "sell", 100.0, 10, 2),
        _order("s1", "spoofy", "buy", 99.0, 400, 6, "cancelled"),
        _order("w1", "washA", "sell", 100.0, 50, 10),
        _order("w2", "washB", "buy", 100.0, 50, 10),
        _order("w3", "washB", "sell", 100.0, 50, 40),
        _order("w4", "washA", "buy", 100.0, 50, 40),
        _order("n1", "noise", "buy", 100.0, 10, 3),
        _order("n2", "noise", "sell", 100.0, 10, 33),
        _order("m1", "mm_0", "buy", 99.5, 20, 1),
        _order("m1", "mm_0", "buy", 99.5, 20, 31, "cancelled"),
    ]
    trades = [
        _trade("t1", 4, 100.0, 10, "noise", "spoofy", "buy"),
        _trade("t2", 10, 100.0, 50, "washB", "washA", "buy"),
        _trade("t3", 40, 100.0, 50, "washA", "washB", "buy"),
        _trade("t4", 33, 100.0, 10, "mm_0", "noise", "sell"),
    ]
    o = pd.DataFrame(orders)
    o["quantity"] = o["quantity"].astype("Int64")
    return pd.DataFrame(trades), o


def test_feature_engine_trader_rows_and_values(tiny_logs):
    trades, orders = tiny_logs
    market, feats = FeatureEngine(CFG).compute(trades, orders)
    assert feats.index.names == ["commodity", "window_start", "trader_id"]
    assert not market.empty
    traders = set(feats.index.get_level_values("trader_id"))
    assert "mm_0" not in traders, "designated market makers are excluded"
    row = feats.xs("spoofy", level="trader_id").iloc[0]
    assert row["spoof_signature"] == 1.0          # large cancel with an opposite-side fill
    assert row["fast_cancel_ratio"] > 0.9          # cancelled after 4 ticks
    assert row["large_cancel_ratio"] == 1.0
    wash = feats.xs("washA", level="trader_id").iloc[0]
    assert wash["top_cpty_share"] == 1.0           # all volume with washB
    assert wash["reciprocal_ratio"] == 1.0         # bought and sold back with the same account
    assert wash["net_flatness"] == 1.0
    assert not feats.isna().any().any(), "features must not contain NaN"


def test_feature_engine_short_run_returns_empty():
    trades = pd.DataFrame([_trade("t1", 1, 100.0, 1, "a", "b")])
    orders = pd.DataFrame([_order("o1", "a", "buy", 100.0, 1, 1)])
    market, feats = FeatureEngine(CFG).compute(trades, orders)
    assert market.empty and feats.empty


def test_trader_level_labels(tiny_logs):
    trades, orders = tiny_logs
    feats = FeatureEngine(CFG).generate_features(trades, orders)
    events = pd.DataFrame([{
        "event_id": "e1", "commodity": "gold", "persona": "wash_trading",
        "start_tick": 0, "end_tick": 59, "trader_ids": ["washA", "washB"], "parameters": {},
    }])
    y = DetectorPipeline(CFG).generate_labels(feats, events)
    pos = y[y["wash_trading"] == 1].index.get_level_values("trader_id")
    assert set(pos) == {"washA", "washB"}
    assert y["spoofing"].sum() == 0


@pytest.fixture(scope="module")
def sim_features():
    out = run_simulation(SimConfig(commodity="gold", n_ticks=2500, write_to_disk=False), seed=5)
    pipe = DetectorPipeline(SIM_CFG)
    X = pipe.feature_engine.generate_features(out.trade_log, out.order_log)
    y = pipe.generate_labels(X, out.manipulation_events)
    return X, y, out.manipulation_events


def test_anomaly_detector_percentile_scores(sim_features):
    X, _, _ = sim_features
    det = AnomalyDetector(CFG)
    det.fit(X)
    s = det.predict_scores(X)
    assert s.between(0, 1).all()
    assert abs(det.predict_labels(X).mean() - 0.05) < 0.03
    expl = det.explain(X.head(3))
    assert len(expl) == 3 and all(isinstance(d, dict) and d for d in expl)


def test_supervised_detector_zero_positive_strict_mode(sim_features):
    X, y, _ = sim_features
    cfg = {**CFG, "supervised": {**CFG["supervised"], "require_all_personas": True}}
    empty = y.copy() * 0
    with pytest.raises(InsufficientLabelsError):
        SupervisedDetector(cfg).fit(X, empty)


def test_pipeline_end_to_end_flags_schema(sim_features, tmp_path):
    X, y, events = sim_features
    pipe = DetectorPipeline(SIM_CFG)
    pipe.train(X, y, save_dir=tmp_path)
    res = pipe.score(X)
    for p in CFG["personas"]:
        assert f"prob_{p}" in res and f"flag_{p}" in res
    flags = pipe.build_flags(X, res)
    io_utils.validate(flags, "flags")                     # FR-2.5 contract
    pipe.write_flags(flags, tmp_path / "flags.parquet")
    back = read_flags(tmp_path / "flags.parquet")
    flagged = back[back["final_flag"]]
    assert all(isinstance(d, dict) and None not in d.values() for d in back["top_shap_features"])
    assert (flagged["top_shap_features"].apply(len) > 0).all(), "every flag carries SHAP evidence"
    trained = set(pipe.supervised_detector.models)
    assert all(set(d) == trained for d in back["persona_scores"]), "every row carries all persona probabilities"
    row = back[back["final_flag"] & back["predicted_persona"].isin(trained)].iloc[0]
    assert row["persona_scores"][row["predicted_persona"]] == pytest.approx(row["final_score"], abs=1e-3)
    assert (tmp_path / "shap_importance.json").exists()
    metrics = compute_round_metrics(res, y, CFG["personas"], events=events)
    for p, m in metrics.items():
        if m["support"]:
            assert m["f1"] > 0.3, f"{p} in-sample F1 sanity floor"


def test_event_detection_latency_nan_for_misses():
    idx = pd.MultiIndex.from_tuples([("gold", 0, "a"), ("gold", 30, "a")], names=["commodity", "window_start", "trader_id"])
    res = pd.DataFrame({"window_end": [59, 89], "flag_spoofing": [False, True]}, index=idx)
    events = pd.DataFrame([
        {"commodity": "gold", "persona": "spoofing", "start_tick": 40, "end_tick": 100, "trader_ids": ["a"]},
        {"commodity": "gold", "persona": "spoofing", "start_tick": 40, "end_tick": 100, "trader_ids": ["b"]},
    ])
    n, det, lats = event_detection(res, events, "spoofing")
    assert (n, det, lats) == (2, 1, [49.0])
    y = pd.DataFrame({"spoofing": [0, 1]}, index=idx)
    m = compute_round_metrics(res, y, ["spoofing"], events=events)["spoofing"]
    assert m["events_detected"] == 1 and m["detection_latency_ticks"] == 49.0
    m_none = compute_round_metrics(res.assign(flag_spoofing=False), y, ["spoofing"], events=events)["spoofing"]
    assert np.isnan(m_none["detection_latency_ticks"])

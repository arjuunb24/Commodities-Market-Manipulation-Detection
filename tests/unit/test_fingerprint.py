"""
tests/unit/test_fingerprint.py
==============================
Manipulator accounts must not be recognisable by their background flow, and an
account's own history must not turn "back to normal" into an alert.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import src.detector.features as F
from src.core.agents.manipulation.common import ManipulatorAccount
from src.core.agents.occasional_trader import OccasionalTrader
from src.core.order_book import BookState
from src.core.simulation import SimConfig, run_simulation


def _state(tick=1):
    return BookState(best_bid=99.9, best_ask=100.1, mid_price=100.0, bid_depth=[(99.9, 50)], ask_depth=[(100.1, 50)],
                     last_trade_price=100.0, tick=tick)


def test_occasional_trader_and_manipulator_background_are_the_same_process():
    occ = OccasionalTrader("occ_0", np.random.default_rng(7), activity_rate=0.05)
    man = ManipulatorAccount("pnd_0", np.random.default_rng(7), background_rate=0.05)
    for a in (occ, man):
        a.tick_size = 0.1
    seq = lambda a: [(o.side, o.price, o.quantity) if o else None for o in (a.decide(_state(t), t) for t in range(400))]
    assert seq(occ) == seq(man)          # same seed, same code path: identical order streams


def test_simulation_includes_occasional_traders():
    cfg = SimConfig(commodity="gold", n_ticks=600, write_to_disk=False, n_occasional_traders=5)
    cfg.enable_spoofing = cfg.enable_wash_trading = cfg.enable_pump_and_dump = cfg.enable_layering = False
    out = run_simulation(cfg, 3)
    occ = out.order_log[out.order_log["trader_id"].str.startswith("occ_")]
    assert occ["trader_id"].nunique() == 5 and len(occ) > 0


def _history(values):
    rows = []
    for w, v in enumerate(values):
        row = {"win": w, "trader_id": "acct"}
        row.update({c: 0.2 for c in F.SELF_Z_COLUMNS})
        row["n_orders"] = 20                     # a normally active account (evidence weighting ~0.8)
        row["cancel_ratio"] = v
        rows.append(row)
    return pd.DataFrame(rows)


def test_return_to_normal_after_a_burst_is_not_flagged(monkeypatch):
    rng = np.random.default_rng(0)
    normal = list(0.30 + rng.normal(0, 0.03, 20))      # sparse history...
    burst = [0.95] * 40                          # ...dominated by a long abnormal stretch (an episode)
    after = list(0.30 + rng.normal(0, 0.03, 20))
    f = _history(normal + burst + after)
    eng = F.FeatureEngine({"feature_engine": {"window": "3min", "step": "30s"}})
    z_trim = eng._add_self_z(f.copy())["self_z_cancel_ratio"].to_numpy()
    monkeypatch.setattr(F, "SELF_Z_TRIM", 1e9)  # old behaviour: the burst joins the baseline
    z_plain = eng._add_self_z(f.copy())["self_z_cancel_ratio"].to_numpy()
    start = len(normal)
    late_burst = slice(start + len(burst) - 10, start + len(burst))
    assert np.abs(z_trim[late_burst]).mean() >= 4                  # a long episode stays visible to its end...
    assert np.abs(z_plain[late_burst]).mean() < 1                  # ...where the old baseline absorbed it
    tail = slice(len(normal) + len(burst) + 8, None)               # back to normal, past the window overlap
    assert np.abs(z_trim[tail]).max() < 1                          # and normal again looks normal
    assert np.abs(z_trim[tail]).max() < np.abs(z_plain[tail]).max()


def _engine():
    return F.FeatureEngine({"feature_engine": {"window": "3min", "step": "30s"}})


def test_activity_baseline_is_not_inflated_by_one_burst(monkeypatch):
    acts = [10] * 20 + [300] * 30 + [10] * 20          # quiet, a long pump burst, quiet again
    f = pd.DataFrame({"win": range(len(acts)), "trader_id": "pnd_0", "net_qty": 0.0,
                      "exec_vol": [a / 2 for a in acts], "n_orders": [a / 2 for a in acts], "exec_side_imbalance": 0.0})
    robust = _engine()._add_temporal(f.copy())["activity_vs_history"].to_numpy()
    monkeypatch.setattr(F, "ACTIVITY_ROBUST", False)
    mean = _engine()._add_temporal(f.copy())["activity_vs_history"].to_numpy()
    after = slice(56, None)                               # back to normal, past the window overlap
    assert np.abs(robust[after]).max() < 0.5              # normal again looks normal
    assert mean[after].min() < -1.5                       # the old mean read it as a collapse in activity


def test_evidence_weighting_damps_sparse_windows():
    rows = []
    for w in range(40):
        row = {"win": w, "trader_id": "acct", **{c: 0.2 for c in F.SELF_Z_COLUMNS}}
        row["n_orders"] = 2 if w == 30 else (60 if w == 35 else 10)
        row["cancel_ratio"] = 0.95 if w in (30, 35) else 0.3
        rows.append(row)
    z = _engine()._add_self_z(pd.DataFrame(rows)).set_index("win")["self_z_cancel_ratio"]
    assert abs(z[30]) < abs(z[35]) * 0.5                  # same jump: 2 orders count far less than 60


def test_ring_features_find_a_rotating_ring_but_not_hft_flow():
    rows = []
    ring = [f"wash_0{c}" for c in "abcdef"]
    for i, a in enumerate(ring):                          # each ring member trades with its two neighbours
        for b in (ring[(i + 1) % 6], ring[(i - 1) % 6]):
            rows.append({"win": 5, "trader_id": a, "cpty": b, "bought": 40.0, "sold": 40.0})
    for k in range(20):                                   # ordinary traders all trade with one HFT
        rows.append({"win": 5, "trader_id": f"noise_{k}", "cpty": "hft_0", "bought": 30.0, "sold": 30.0})
        rows.append({"win": 5, "trader_id": "hft_0", "cpty": f"noise_{k}", "bought": 30.0, "sold": 30.0})
    pw = pd.DataFrame(rows)
    f = pd.DataFrame({"win": 5, "trader_id": ring + [f"noise_{k}" for k in range(20)] + ["hft_0"]})
    out = _engine()._add_ring_features(f, pw, median_size=10.0).set_index("trader_id")
    assert (out.loc[ring, "ring_size"] == 6).all() and (out.loc[ring, "ring_volume_share"] > 0.99).all()
    assert (out.loc[[f"noise_{k}" for k in range(20)], "mutual_cpty_share"] == 0).all()   # HFT is not a ring partner
    assert out.loc["hft_0", "ring_size"] == 1

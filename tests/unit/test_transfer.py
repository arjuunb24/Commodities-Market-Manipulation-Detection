"""Randomised persona variants, the fake-depth family classifier and the anomaly score as an input."""

from __future__ import annotations

import numpy as np
import pytest
import yaml

from src.adversarial.variants import VARIANT_LIMITS, random_variant, variant_config
from src.agentic.tools import PARAM_BOUNDS
from src.core.persona_params import resolve_persona_config
from src.core.simulation import SimConfig, run_simulation
from src.detector.pipeline import DetectorPipeline

PERSONAS = ["spoofing", "wash_trading", "pump_and_dump", "layering"]
CFG = {
    "feature_engine": {"window": "3min", "step": "30s", "exclude_trader_prefixes": ["mm_"]},
    "isolation_forest": {"contamination": 0.05, "n_estimators": 50, "random_state": 0},
    "xgboost": {"n_estimators_small": 40, "max_depth_small": 3, "n_estimators": 40},
    "supervised": {"min_positives": 3, "cv_folds": 2},
    "personas": PERSONAS,
}


def _base() -> dict:
    return resolve_persona_config(yaml.safe_load(open("configs/persona_config.yaml")) or {})


def test_random_variant_redraws_a_few_knobs_inside_bounds():
    base = _base()
    v = random_variant(base, np.random.default_rng(1), PERSONAS)
    assert v == random_variant(base, np.random.default_rng(1), PERSONAS)       # reproducible
    for p in PERSONAS:
        changed = [k for k in PARAM_BOUNDS[p] if v[p].get(k) != base[p].get(k)]
        assert 1 <= len(changed) <= 6
        for k in changed:
            typ, lo, hi, _ = PARAM_BOUNDS[p][k]
            hi = min(hi, VARIANT_LIMITS.get(p, {}).get(k, hi))
            assert lo <= v[p][k] <= hi and isinstance(v[p][k], typ)


def test_variant_config_is_a_persona_config_document():
    doc = variant_config(_base(), np.random.default_rng(2), PERSONAS)
    active = resolve_persona_config(doc)
    assert set(PERSONAS) <= set(active)
    assert all(active[p].get("enabled", True) for p in PERSONAS)


@pytest.fixture(scope="module")
def fake_depth_market():
    cfg = SimConfig(commodity="gold", n_ticks=3000, write_to_disk=False, enable_layering=True,
                    enable_wash_trading=False, enable_pump_and_dump=False,
                    spoof_start_tick=300, spoof_end_tick=1200, layer_start_tick=1500, layer_end_tick=2500)
    out = run_simulation(cfg, seed=4)
    pipe = DetectorPipeline(CFG)
    X = pipe.feature_engine.generate_features(out.trade_log, out.order_log)
    y = pipe.generate_labels(X, out.manipulation_events, out.order_log)
    return X, y


def test_family_classifier_flags_reach_a_member_persona(fake_depth_market):
    X, y = fake_depth_market
    pipe = DetectorPipeline(CFG)
    pipe.train(X, y)
    assert "fake_depth" in pipe.supervised_detector.models
    res = pipe.score(X)
    fam = res["prob_fake_depth"] >= pipe.supervised_detector.thresholds["fake_depth"]
    assert fam.any()
    assert (res.loc[fam, "flag_spoofing"] | res.loc[fam, "flag_layering"]).all()
    assert res.loc[fam, "predicted_persona"].isin(["spoofing", "layering"]).all()


def test_anomaly_score_is_a_classifier_input_only_when_enabled(fake_depth_market):
    X, y = fake_depth_market
    on = DetectorPipeline(CFG)
    on.train(X, y)
    assert "iso_score" in on.supervised_detector.feature_columns
    off = DetectorPipeline({**CFG, "supervised": {**CFG["supervised"], "iso_score_feature": False,
                                                   "family_models": {}}})
    off.train(X, y)
    assert "iso_score" not in off.supervised_detector.feature_columns
    assert "fake_depth" not in off.supervised_detector.models
    assert set(off.score(X).columns) >= {"iso_score", "final_flag"}


def test_family_prf_counts_sibling_label_as_correct():
    import pandas as pd
    from src.adversarial.metrics import family_prf
    idx = pd.RangeIndex(4)
    y = pd.DataFrame({"spoofing": [1, 0, 0, 0], "layering": [0, 1, 0, 0]}, index=idx)
    res = pd.DataFrame({"final_flag": [True, True, True, False],
                        "predicted_persona": ["layering", "spoofing", "spoofing", "none"]}, index=idx)
    m = family_prf(res, y, ("spoofing", "layering"))
    # both episodes caught under the sibling label (counted), one false alarm on row 2
    assert m["family_recall"] == 1.0 and abs(m["family_precision"] - 2 / 3) < 1e-3

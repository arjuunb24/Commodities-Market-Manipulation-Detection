"""
tests/unit/test_improvements.py
===============================
Self-history features, corroborated anomaly alerts, and the strategist's
text-proposal parsing and candidate probing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import yaml

from src.detector.features import SELF_Z_COLUMNS, FeatureEngine


def _engine():
    return FeatureEngine({"feature_engine": {"window": "3min", "step": "30s"}})


def _frame(sizes_by_trader: dict[str, list[float]]) -> pd.DataFrame:
    rows = []
    for t, sizes in sizes_by_trader.items():
        for w, s in enumerate(sizes):
            row = {"win": w, "trader_id": t}
            row.update({c: 0.1 for c in SELF_Z_COLUMNS})
            row["max_order_size_rel"] = s
            rows.append(row)
    return pd.DataFrame(rows)


def test_self_z_is_quiet_for_consistent_accounts_and_loud_for_changes():
    n = 60
    algo = list(np.random.default_rng(0).uniform(8, 12, n))      # always large: normal for itself
    switch = [1.0] * 40 + [30.0] * 20                           # small trader that starts spoofing
    out = _engine()._add_self_z(_frame({"algo_0": algo, "spoofer_0": switch}))
    z = out.set_index(["trader_id", "win"])["self_z_max_order_size_rel"]
    assert z.loc["algo_0"].abs().max() < 4
    assert z.loc["spoofer_0"].iloc[45] > 10
    assert z.loc["spoofer_0"].iloc[:5].eq(0).all()               # no history yet -> 0


def test_uncorroborated_isolation_forest_flag_is_review_only():
    from src.detector.pipeline import DetectorPipeline
    cfg = yaml.safe_load(open("configs/detector_config.yaml"))
    pipe = DetectorPipeline(cfg)

    class _Iso:
        def predict_scores(self, X): return pd.Series([0.995, 0.995, 0.2], index=X.index)
        def predict_labels(self, X): return pd.Series([True, True, False], index=X.index)

    class _Sup:
        thresholds = {p: 0.8 for p in cfg["personas"]}
        models = {}
        def predict_proba(self, X):
            return pd.DataFrame({p: [0.5 if p == "spoofing" else 0.0, 0.0, 0.0] for p in cfg["personas"]}, index=X.index)
        def predict_flags(self, probs): return probs >= 0.8

    idx = pd.MultiIndex.from_tuples([("gold", 0, "a"), ("gold", 0, "b"), ("gold", 0, "c")],
                                    names=["commodity", "window_start", "trader_id"])
    X = pd.DataFrame({"window_end": [179] * 3, **{f"self_z_{c}": [0.0, 0.0, 0.0] for c in SELF_Z_COLUMNS}}, index=idx)
    pipe.anomaly_detector, pipe.supervised_detector = _Iso(), _Sup()
    pipe._with_context = lambda X: X.assign(mkt_anomaly_score=0.0)
    pipe.is_trained = True
    res = pipe.score(X)
    # a: forest + spoofing prob 0.5 >= 0.5 x threshold 0.8 -> corroborated alert; b: forest only -> review
    assert list(res["alert_tier"]) == ["alert", "review", "none"]
    assert list(res["final_flag"]) == [True, False, False]


def test_proposals_written_as_text_are_parsed():
    from src.agentic.adversarial_strategist import _proposals_in_text
    text = 'Here it is:\n{\n "persona": "wash_trading",\n "parameters": {"n_groups": 5},\n "reasoning": "x"\n} done'
    got = _proposals_in_text(text)
    assert got == [{"persona": "wash_trading", "parameters": {"n_groups": 5}, "reasoning": "x"}]
    assert _proposals_in_text("no json here {not json}") == []


def test_alternatives_validated_and_best_probe_candidate_kept(tmp_path):
    import json
    from src.agentic.adversarial_strategist import AdversarialStrategist
    from src.agentic.tools import ToolRegistry
    run = tmp_path / "run"; (run / "round_0").mkdir(parents=True)
    (run / "round_0" / "metrics.json").write_text(json.dumps({"params_used": {"spoofing": {"n_accounts": 1, "frequency": 0.4}}}))
    cfg = tmp_path / "persona_config.yaml"; cfg.write_text(yaml.dump({"spoofing": {"n_accounts": 1, "frequency": 0.4}}))
    tools = ToolRegistry(run_dir=run, persona_config_path=cfg)
    res = tools.propose_mutation("spoofing", {"n_accounts": 2}, "r",
                                 alternatives=[{"frequency": 0.1}, {"n_accounts": 99}])   # second is out of bounds
    assert res["alternatives_accepted"] == 1 and len(tools.candidates["spoofing"]) == 2

    strat = AdversarialStrategist({"max_tool_calls": 1}, run_dir=run, tools=tools,
                                  probe_fn=lambda cfgs: [{"spoofing": {"score": s}}
                                                         for s in ([0.1, 0.6] + [0.0] * (len(cfgs) - 2))])
    log = {}
    out = strat._probe_candidates(0, ["spoofing"], {"spoofing": "agentic"}, log)
    assert out["spoofing"]["chosen"] == 1
    written = yaml.safe_load(cfg.read_text())["spoofing"]
    assert written["frequency"] == 0.1 and written["n_accounts"] == 1        # alternative applied on the base params
    assert "[probe]" in log["spoofing"]


def test_opposite_fill_flags_match_cross_join():
    from src.detector.features import _opposite_fill_flags
    rng = np.random.default_rng(3)
    n, m = 400, 3000
    lc = pd.DataFrame({"trader_id": rng.choice(["a", "b", "c", "d"], n), "side": rng.choice(["buy", "sell"], n),
                       "placed_tick": rng.integers(0, 5000, n)})
    lc["tick"] = lc["placed_tick"] + rng.integers(0, 30, n)
    lc = lc.reset_index().rename(columns={"index": "cid"})
    legs = pd.DataFrame({"trader_id": rng.choice(["a", "b", "c", "e"], m), "tick": rng.integers(0, 5100, m)})
    is_buy = rng.random(m) < 0.5
    legs["buy"], legs["sell"] = np.where(is_buy, 10, 0), np.where(is_buy, 0, 10)
    # reference: the original cross join
    j = lc.merge(legs.rename(columns={"tick": "fill_tick"}), on="trader_id")
    j = j[(j["fill_tick"] >= j["placed_tick"]) & (j["fill_tick"] <= j["tick"] + 2)]
    j = j[((j["side"] == "buy") & (j["sell"] > 0)) | ((j["side"] == "sell") & (j["buy"] > 0))]
    expected = lc["cid"].isin(j["cid"].unique()).astype(float).to_numpy()
    got = _opposite_fill_flags(lc, legs)
    assert expected.sum() > 50 and (got == expected).all()


def test_model_proposal_shapes_are_normalised(tmp_path):
    """Shapes seen from the model in a real run: nested alternatives and flattened knobs."""
    from src.agentic.adversarial_strategist import _normalise_proposal
    from src.agentic.tools import ToolRegistry
    cfg = tmp_path / "persona_config.yaml"; cfg.write_text(yaml.dump({"spoofing": {"n_accounts": 1}}))
    tools = ToolRegistry(run_dir=tmp_path, persona_config_path=cfg)
    flat = {"persona": "spoofing", "n_split_orders": 1, "size_jitter": 0.6, "reasoning": "r",
            "alternatives": [{"parameters": {"cancel_jitter": 0.5}, "reasoning": "x"}, {"n_accounts": 2}]}
    res = tools.propose_mutation(**_normalise_proposal(flat))
    assert "error" not in res and res["alternatives_accepted"] == 2
    assert tools.candidates["spoofing"] == [{"n_split_orders": 1, "size_jitter": 0.6}, {"cancel_jitter": 0.5}, {"n_accounts": 2}]


def _pnd_round(run, r, params, ratio, score):
    import json
    d = run / f"round_{r}"; d.mkdir(parents=True)
    (d / "metrics.json").write_text(json.dumps({
        "round": r, "params_used": {"pump_and_dump": params},
        "effectiveness": {"pump_and_dump": {"ignition_pressure": 0.07 * ratio, "primary_metric": "ignition_pressure"}},
        "effectiveness_ratio": {"pump_and_dump": ratio}, "mutation_score": {"pump_and_dump": score}}))


def test_exploration_candidate_spends_effect_surplus(tmp_path):
    from src.agentic.adversarial_strategist import AdversarialStrategist
    from src.agentic.tools import ToolRegistry
    run = tmp_path / "run"
    base = {"ignition_aggression": 0.002, "accumulation_size": 150, "n_coordinated_accounts": 3, "burst_duration_ticks": 200}
    _pnd_round(run, 0, base, 1.0, None)
    _pnd_round(run, 1, base, 2.5, 0.2)                     # surplus: 2.5x of the round-0 effect
    tools = ToolRegistry(run_dir=run, persona_config_path=tmp_path / "none.yaml")
    strat = AdversarialStrategist({"max_tool_calls": 1}, run_dir=run, tools=tools)
    ex = strat._exploration_candidate("pump_and_dump", base)
    # the first two effect-costing knobs move 40% towards their effect-lowering bound
    assert ex["ignition_aggression"] == pytest.approx(0.002 + 0.4 * (0.0002 - 0.002))
    assert ex["accumulation_size"] == round(150 + 0.4 * (50 - 150))


def test_last_probe_lines_mark_failed_and_kept(tmp_path):
    import json
    from src.agentic.adversarial_strategist import AdversarialStrategist
    from src.agentic.tools import ToolRegistry
    run = tmp_path / "run"; d = run / "round_2"; d.mkdir(parents=True)
    (d / "metrics.json").write_text("{}")
    (d / "strategist.json").write_text(json.dumps({"round": 2, "personas": {"layering": {"probe": {"chosen": 1, "candidates": [
        {"changes": {"layer_spacing": 0.0004}, "coverage": 0.82, "effect_ratio": 0.43, "score": 0.0},
        {"changes": {"n_layers": 3}, "coverage": 0.97, "effect_ratio": 0.67, "score": 0.03}]}}}}))
    strat = AdversarialStrategist({"max_tool_calls": 1}, run_dir=run, tools=ToolRegistry(run_dir=run))
    lines = strat._last_probe(2, "layering")
    assert "FAILED" in lines[0] and lines[1].lstrip().startswith("*")

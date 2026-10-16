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


def test_probe_rejects_candidates_the_probe_puts_at_risk(tmp_path):
    import json
    from src.agentic.adversarial_strategist import AdversarialStrategist
    from src.agentic.tools import ToolRegistry
    run = tmp_path / "run"; (run / "round_0").mkdir(parents=True)
    (run / "round_0" / "metrics.json").write_text(json.dumps({"params_used": {"spoofing": {"n_accounts": 1, "frequency": 0.4}}}))
    cfg = tmp_path / "persona_config.yaml"; cfg.write_text(yaml.dump({"spoofing": {"n_accounts": 1, "frequency": 0.4}}))
    tools = ToolRegistry(run_dir=run, persona_config_path=cfg)
    tools.propose_mutation("spoofing", {"n_accounts": 2}, "r", alternatives=[{"frequency": 0.3}])
    # primary has the best score but the probe puts its effect at 0.52 (< 0.65): the safe alternative wins
    res = [{"score": 0.09, "effect_ratio": 0.52}, {"score": 0.05, "effect_ratio": 0.8}]
    strat = AdversarialStrategist({"max_tool_calls": 1}, run_dir=run, tools=tools,
                                  probe_fn=lambda cfgs: [{"spoofing": res[min(k, 1)]} for k in range(len(cfgs))])
    out = strat._probe_candidates(0, ["spoofing"], {"spoofing": "agentic"}, {})
    assert out["spoofing"]["chosen"] == 1


def test_weak_coverage_counts_as_missed():
    from src.adversarial.effectiveness import evasion_components
    base = {"caught_coverage": 0.6, "events_total": 16, "caught_events_detected": 16, "caught_latency_ticks": 30.0}
    assert evasion_components(base)["missed"] == 0.0
    assert evasion_components({**base, "caught_weak_share": 0.25})["missed"] == 0.25


def test_threshold_plateau_centre_and_sticky_previous():
    from src.detector.classifiers import SupervisedDetector
    det = SupervisedDetector({"personas": ["spoofing"]})
    grid = np.linspace(0.25, 0.9, 27)
    scores = [0.80] * 27
    scores[3] = 0.86                                 # lone noisy spike
    for i in range(12, 19): scores[i] = 0.85          # broad plateau
    t = det._pick_threshold(grid, scores, None)
    assert abs(t - grid[15]) < 1e-9                   # centre of the plateau, not the spike
    assert det._pick_threshold(grid, scores, grid[13]) == grid[13]     # previous still near-best: kept
    assert det._pick_threshold(grid, scores, 0.9) == t                 # previous clearly worse: replaced


def test_probe_baseline_uses_round0_params_and_is_cached(tmp_path):
    import json
    from src.adversarial.orchestrator import RoundOrchestrator
    run = tmp_path / "run"; (run / "round_0").mkdir(parents=True)
    (run / "round_0" / "metrics.json").write_text(json.dumps({"params_used": {"wash_trading": {"trade_frequency": 0.5}}}))
    orch = RoundOrchestrator(run, yaml.safe_load(open("configs/detector_config.yaml")),
                             persona_config_path=tmp_path / "none.yaml")
    seen = []
    def fake(active, root, features=True):
        seen.append(active["wash_trading"]["trade_frequency"])
        return None, None, None, {"wash_trading": {"wash_volume_share": 0.06, "primary_metric": "wash_volume_share"}}
    orch._simulate_probe_set = fake
    base = {"wash_trading": {"trade_frequency": 0.3}}
    assert orch._probe_baseline(base, tmp_path / "p")["wash_trading"]["wash_volume_share"] == 0.06
    assert seen == [0.5]                                   # simulated with the ROUND-0 value, not the current one
    orch._probe_baseline(base, tmp_path / "p")
    assert seen == [0.5]                                   # second call reads probe_baseline.json


def test_truncated_replies_do_not_use_up_persona_turns(tmp_path, monkeypatch):
    import json
    import src.agentic.adversarial_strategist as strat_mod
    from src.agentic.adversarial_strategist import AdversarialStrategist
    from src.agentic.tools import ToolRegistry
    personas = ["spoofing", "wash_trading", "pump_and_dump", "layering"]
    params = {"spoofing": {"n_accounts": 2}, "wash_trading": {"n_groups": 2},
              "pump_and_dump": {"n_coordinated_accounts": 4}, "layering": {"n_layers": 4}}
    run = tmp_path / "run"; (run / "round_0").mkdir(parents=True)
    (run / "round_0" / "metrics.json").write_text(json.dumps({"params_used": {}}))
    cfg = tmp_path / "persona_config.yaml"; cfg.write_text(yaml.dump({}))
    tools = ToolRegistry(run_dir=run, persona_config_path=cfg)
    loop = {"finish_reason": "length", "message": {"content": "Now propose for wash_trading. " * 50}}
    def call(p):
        return {"finish_reason": "tool_calls", "message": {"role": "assistant", "content": None, "tool_calls": [
            {"id": p, "type": "function", "function": {"name": "propose_mutation",
             "arguments": json.dumps({"persona": p, "parameters": params[p], "reasoning": "r"})}}]}}
    replies = iter([loop, call("spoofing"), loop, call("wash_trading"), call("pump_and_dump"), call("layering")])

    class _Client:
        def complete(self, **kw): return {"choices": [next(replies)]}

    monkeypatch.setattr(strat_mod, "llm_unavailable_reason", lambda cfg: None)
    strat = AdversarialStrategist({"max_tool_calls": 5}, run_dir=run, tools=tools)
    strat.client = _Client()
    monkeypatch.setattr(strat, "_build_context", lambda *a, **k: "ctx")
    out = strat.run_strategist_round(0, personas)
    assert out["source"] == {p: "agentic" for p in personas}     # 2 cut-off replies + 4 calls fit in 5 turns


def test_staggered_layering_never_places_a_multi_level_batch_in_one_tick():
    from src.core.agents.manipulation.layering import LayeringAgent
    from src.core.order_book import BookState
    state = BookState(tick=0, best_bid=100.0, best_ask=100.02, mid_price=100.01,
                      bid_depth=[(100.0, 50), (99.99, 50)], ask_depth=[(100.02, 50), (100.03, 50)],
                      last_trade_price=100.01)
    agent = LayeringAgent("ly", np.random.default_rng(1), n_layers=4, frequency=1.0, cancellation_delay_ticks=5,
                          placement_stagger=2)
    layers_per_tick = []
    for t in range(40):
        orders = agent.manipulate(state, t)
        layers_per_tick.append(sum(o.order_id.endswith("l") for o in orders))
    assert max(layers_per_tick) == 1 and sum(layers_per_tick) >= 8      # one layer per tick, cycles still complete
    flat = LayeringAgent("ly", np.random.default_rng(1), n_layers=4, frequency=1.0, cancellation_delay_ticks=5)
    assert sum(o.order_id.endswith("l") for o in flat.manipulate(state, 0)) == 4   # stagger 0: unchanged ladder


def test_self_history_corroboration_needs_two_features():
    from src.detector.pipeline import DetectorPipeline
    cfg = yaml.safe_load(open("configs/detector_config.yaml"))
    pipe = DetectorPipeline(cfg)

    class _Iso:
        def predict_scores(self, X): return pd.Series(0.995, index=X.index)
        def predict_labels(self, X): return pd.Series(True, index=X.index)

    class _Sup:
        thresholds = {p: 0.8 for p in cfg["personas"]}
        models = {}
        def predict_proba(self, X): return pd.DataFrame({p: 0.0 for p in cfg["personas"]}, index=X.index)
        def predict_flags(self, probs): return probs >= 0.8

    idx = pd.MultiIndex.from_tuples([("gold", 0, "algo_0"), ("gold", 0, "x")], names=["commodity", "window_start", "trader_id"])
    z = {f"self_z_{c}": [0.0, 0.0] for c in SELF_Z_COLUMNS}
    z["self_z_max_order_size_rel"] = [20.0, 20.0]          # one feature jumps (new parent order) ...
    z["self_z_n_orders"] = [0.0, 12.0]                       # ... vs. two features deviating at once
    X = pd.DataFrame({"window_end": [179, 179], **z}, index=idx)
    pipe.anomaly_detector, pipe.supervised_detector = _Iso(), _Sup()
    pipe._with_context = lambda X: X.assign(mkt_anomaly_score=0.0)
    pipe.is_trained = True
    assert list(pipe.score(X)["alert_tier"]) == ["review", "alert"]


def test_family_coverage_counts_sibling_label():
    from src.adversarial.metrics import compute_round_metrics
    idx = pd.MultiIndex.from_tuples([("gold", w, "spoofer_0") for w in (0, 30, 60)],
                                    names=["commodity", "window_start", "trader_id"])
    res = pd.DataFrame({"window_end": [179, 209, 239], "flag_spoofing": False, "flag_layering": [True, True, False],
                        "final_flag": [True, True, False]}, index=idx)
    y = pd.DataFrame({"spoofing": 1, "layering": 0}, index=idx)
    ev = pd.DataFrame([{"event_id": "e", "persona": "spoofing", "commodity": "gold", "trader_ids": ["spoofer_0"],
                        "start_tick": 0, "end_tick": 300}])
    m = compute_round_metrics(res, y, ["spoofing", "layering"], events=ev)["spoofing"]
    assert m["episode_coverage"] == 0.0 and m["family_coverage"] == pytest.approx(2 / 3, abs=1e-3)


def test_probe_effect_is_anchored_on_last_real_round(tmp_path):
    import json
    from src.adversarial.orchestrator import RoundOrchestrator
    run = tmp_path / "run"
    for r, ratio in ((0, 1.0), (2, 0.9)):
        (run / f"round_{r}").mkdir(parents=True)
        (run / f"round_{r}" / "metrics.json").write_text(json.dumps({
            "params_used": {"layering": {"n_layers": 4 if r == 0 else 3}}, "effectiveness_ratio": {"layering": ratio}}))
    orch = RoundOrchestrator(run, yaml.safe_load(open("configs/detector_config.yaml")), persona_config_path=tmp_path / "none.yaml")
    orch.pipeline.is_trained = True
    depth = {4: 0.5, 3: 0.4, 2: 0.2}                        # probe-market fake depth by n_layers
    def fake(active, root, features=True):
        e = {"layering": {"fake_depth_share": depth[active["layering"]["n_layers"]], "primary_metric": "fake_depth_share"}}
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), e
    orch._simulate_probe_set = fake
    orch.pipeline.score = lambda X: pd.DataFrame()
    import src.adversarial.orchestrator as om
    om_compute = om.compute_round_metrics
    om.compute_round_metrics = lambda *a, **k: {}
    try:
        out = orch.probe([{"layering": {"n_layers": 2}}], round_num=2)
    finally:
        om.compute_round_metrics = om_compute
    # anchored: real ratio 0.9 at n_layers=3 x candidate / reference on the probe markets (0.2 / 0.4) = 0.45;
    # direct: candidate / round-0 probe baseline = 0.2 / 0.5 = 0.4; the estimate is their mean
    d = out[0]["layering"]["effect_detail"]
    assert d["anchored"] == pytest.approx(0.45) and d["direct"] == pytest.approx(0.4)
    assert out[0]["layering"]["effect_ratio"] == pytest.approx(0.425)


def test_kept_threshold_is_released_after_max_keep_rounds():
    from src.detector.classifiers import SupervisedDetector
    det = SupervisedDetector({"personas": ["spoofing"]})
    grid = np.linspace(0.25, 0.9, 27)
    scores = [0.80] * 27
    for i in range(12, 19): scores[i] = 0.85
    prev = grid[13]                                      # near-best: normally kept
    assert det._pick_threshold(grid, scores, prev, kept=0) == prev
    assert det._pick_threshold(grid, scores, prev, kept=1) == prev
    assert det._pick_threshold(grid, scores, prev, kept=2) == grid[15]   # released to the plateau centre

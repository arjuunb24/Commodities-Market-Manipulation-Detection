"""
tests/unit/test_effectiveness.py
================================
Manipulation effectiveness accounting, the strategist objective, new evasion
knobs and the persona_config → SimConfig mapping.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.adversarial.effectiveness import (
    EFFECTIVENESS_FLOOR,
    effectiveness_ratio,
    episode_effectiveness,
    mutation_score,
    summarize,
)
from src.core.agents.manipulation.spoofing import SpoofingAgent
from src.core.order_book import BookState
from src.core.persona_params import apply_persona_config, effective_params, resolve_persona_config
from src.core.simulation import SimConfig, run_simulation


def _eff(**kw) -> dict:
    rows = []
    for seed in (1, 2):
        out = run_simulation(SimConfig(commodity="crude_oil_wti", n_ticks=4000, write_to_disk=False,
                                       enable_layering=True, **kw), seed)
        rows.append(episode_effectiveness(out.manipulation_events, out.trade_log, out.order_log))
    return summarize(pd.concat(rows, ignore_index=True))


@pytest.fixture(scope="module")
def baseline_eff():
    return _eff()


def test_effectiveness_detects_neutered_manipulation(baseline_eff):
    neutered = _eff(spoof_aggressiveness=0.008, wash_frequency=0.05, pnd_n_accounts=2,
                    pnd_accumulation_size=50, pnd_burst_duration=900,
                    layer_spacing=0.003, layer_size_multiplier=0.5)
    for persona in ("spoofing", "wash_trading", "pump_and_dump", "layering"):
        ratio = effectiveness_ratio(neutered.get(persona), baseline_eff.get(persona), persona)
        assert ratio is not None and ratio < EFFECTIVENESS_FLOOR, (persona, ratio)


def test_baseline_manipulations_have_an_effect(baseline_eff):
    assert baseline_eff["spoofing"]["fake_depth_share"] > 0.3
    assert baseline_eff["layering"]["fake_depth_share"] > 0.3
    assert baseline_eff["spoofing"]["genuine_fill_rate"] > 0.5
    assert baseline_eff["wash_trading"]["wash_volume_share"] > 0.1
    assert baseline_eff["pump_and_dump"]["ignition_pressure"] > 0.05


def test_mutation_score_rewards_only_effective_evasion():
    pre = {"episode_coverage": 0.4, "events_total": 10, "events_detected": 8, "detection_latency_ticks": 150.0}
    evasion = 0.5 * 0.6 + 0.3 * 0.2 + 0.2 * 0.5
    assert mutation_score(pre, 1.0) == pytest.approx(evasion)
    assert mutation_score(pre, 0.7) == pytest.approx(evasion * 0.7)
    assert mutation_score(pre, 3.0) == pytest.approx(evasion)        # effect above round 0 earns nothing
    assert mutation_score(pre, 0.3) == 0.0                            # "evasion" by not manipulating
    assert mutation_score(None, 1.0) is None                          # no frozen detector in round 0
    all_missed = {"episode_coverage": 0.0, "events_total": 4, "events_detected": 0}
    assert mutation_score(all_missed, 1.0) == pytest.approx(1.0)      # complete escape
    assert mutation_score(0.2, 1.0) == pytest.approx(0.5 * 0.8)       # legacy: bare recall


def test_wash_groups_logged_as_one_episode():
    out = run_simulation(SimConfig(commodity="gold", n_ticks=1500, write_to_disk=False,
                                   enable_spoofing=False, enable_pump_and_dump=False,
                                   wash_start_tick=100, wash_end_tick=900, wash_n_pairs=3, wash_ring_size=3), 3)
    ev = out.manipulation_events
    assert len(ev) == 1 and len(ev.iloc[0]["trader_ids"]) == 9


def test_cross_account_spoofing_uses_different_accounts():
    agent = SpoofingAgent("sp", np.random.default_rng(0), frequency=1.0, n_accounts=2, n_split_orders=3,
                          start_tick=0, end_tick=100)
    state = BookState(best_bid=99.0, best_ask=101.0, mid_price=100.0, bid_depth=[(99.0, 40)],
                      ask_depth=[(101.0, 40)], last_trade_price=100.0, tick=5)
    orders = agent.decide_batch(state, 5)
    genuine = [o for o in orders if o.order_id.endswith("g")]
    spoofs = [o for o in orders if o.order_id.endswith("s")]
    assert len(genuine) == 1 and len(spoofs) == 3
    assert {o.trader_id for o in spoofs} != {genuine[0].trader_id}
    assert len({o.price for o in spoofs}) == 3      # consecutive levels


def test_persona_param_mapping_roundtrip():
    cfg = {"round_0": {"spoofing": {"enabled": True, "frequency": 0.4, "n_accounts": 1},
                       "wash_trading": {"enabled": False}},
           "spoofing": {"n_accounts": 2, "size_jitter": 0.3}}
    active = resolve_persona_config(cfg)
    sim = apply_persona_config(SimConfig(write_to_disk=False), active)
    assert sim.spoof_n_accounts == 2 and sim.spoof_size_jitter == 0.3 and sim.spoof_frequency == 0.4
    assert sim.enable_wash_trading is False
    assert effective_params(active, "spoofing") == {"frequency": 0.4, "n_accounts": 2, "size_jitter": 0.3}


def test_enabled_flag_can_turn_a_persona_on():
    cfg = {"round_0": {"layering": {"enabled": True}, "spoofing": {"enabled": False}}}
    sim = apply_persona_config(SimConfig(write_to_disk=False), resolve_persona_config(cfg))
    assert sim.enable_layering is True and sim.enable_spoofing is False


def test_layering_present_with_repo_baseline_config():
    import yaml
    from pathlib import Path
    cfg = yaml.safe_load((Path(__file__).resolve().parents[2] / "configs" / "persona_config.yaml").read_text())
    sim = apply_persona_config(SimConfig(commodity="gold", n_ticks=3000, write_to_disk=False), resolve_persona_config(cfg))
    out = run_simulation(sim, 4)
    assert "layering" in set(out.manipulation_events["persona"])


def test_evasion_counts_any_alert_not_own_label():
    from src.adversarial.metrics import compute_round_metrics
    idx = pd.MultiIndex.from_tuples([("gold", 0, "sp1"), ("gold", 30, "sp1"), ("gold", 0, "mm")],
                                    names=["commodity", "window_start", "trader_id"])
    results = pd.DataFrame({"window_end": [180, 210, 180], "flag_spoofing": [False, False, False],
                            "flag_layering": [True, True, False], "final_flag": [True, True, False]}, index=idx)
    y = pd.DataFrame({"spoofing": [1, 1, 0], "layering": [0, 0, 0]}, index=idx)
    events = pd.DataFrame([{"persona": "spoofing", "commodity": "gold", "trader_ids": ["sp1"],
                            "start_tick": 0, "end_tick": 200}])
    m = compute_round_metrics(results, y, ["spoofing"], events=events)["spoofing"]
    assert m["episode_coverage"] == 0.0 and m["events_detected"] == 0       # own classifier missed it ...
    assert m["caught_coverage"] == 1.0 and m["caught_events_detected"] == 1  # ... but it was still caught
    assert mutation_score(m, 1.0) == pytest.approx(0.2 * 180 / 300)         # only the delay term: relabelling is not evasion


def _write_round(run_dir, r, params, ratio, score):
    import json
    d = run_dir / f"round_{r}"
    d.mkdir(parents=True)
    (d / "metrics.json").write_text(json.dumps({
        "round": r, "params_used": {"layering": params},
        "effectiveness": {"layering": {"fake_depth_share": 0.5 * ratio, "primary_metric": "fake_depth_share"}},
        "effectiveness_ratio": {"layering": ratio}, "mutation_score": {"layering": score}}))


def test_effect_recovery_resets_to_best_effective_round(tmp_path):
    import yaml
    from src.agentic.tools import ToolRegistry
    run = tmp_path / "run"
    base = {"layer_spacing": 0.0002, "n_layers": 4, "cancel_jitter": 0.0}
    _write_round(run, 0, base, 1.0, None)
    _write_round(run, 1, {**base, "cancel_jitter": 0.5}, 0.9, 0.2)
    _write_round(run, 2, {**base, "cancel_jitter": 0.5, "layer_spacing": 0.001, "n_layers": 3}, 0.42, 0.0)
    cfg = tmp_path / "persona_config.yaml"
    cfg.write_text(yaml.dump({"layering": {**base, "cancel_jitter": 0.5, "layer_spacing": 0.001, "n_layers": 3}}))
    tools = ToolRegistry(run_dir=run, persona_config_path=cfg)

    st = tools.recover_effect("layering")
    assert st["status"] == "failed" and st["anchor_round"] == 1
    assert st["changes_since_anchor"] == {"layer_spacing": [0.0002, 0.001], "n_layers": [4, 3]}
    written = yaml.safe_load(cfg.read_text())["layering"]
    assert written["layer_spacing"] == 0.0002 and written["n_layers"] == 4 and written["cancel_jitter"] == 0.5


def test_effect_at_risk_warns_without_reset(tmp_path):
    import yaml
    from src.agentic.tools import ToolRegistry
    run = tmp_path / "run"
    base = {"layer_spacing": 0.0002, "n_layers": 4}
    _write_round(run, 0, base, 1.0, None)
    _write_round(run, 1, {**base, "n_layers": 3}, 0.6, 0.1)
    cfg = tmp_path / "persona_config.yaml"
    cfg.write_text(yaml.dump({"layering": {**base, "n_layers": 3}}))
    tools = ToolRegistry(run_dir=run, persona_config_path=cfg)
    assert tools.effect_status("layering")["status"] == "at_risk"
    assert tools.recover_effect("layering") is None
    assert yaml.safe_load(cfg.read_text())["layering"]["n_layers"] == 3


def test_effect_guard_reverts_ignored_effect_costing_changes(tmp_path):
    """guard_v1 round 4: layering at risk (0.59x) after spacing 0.0002->0.001 and 4->2 layers;
    the model only touched cancel timing, so the guard restores spacing and layer count."""
    import yaml
    from src.agentic.tools import ToolRegistry
    run = tmp_path / "run"
    r2 = {"layer_spacing": 0.0002, "n_layers": 4, "layer_size_multiplier": 2.0, "cancel_jitter": 0.7}
    r3 = {"layer_spacing": 0.001, "n_layers": 2, "layer_size_multiplier": 3.0, "cancel_jitter": 0.8}
    _write_round(run, 0, {**r2, "cancel_jitter": 0.0}, 1.0, None)
    _write_round(run, 2, r2, 0.75, 0.05)
    _write_round(run, 3, r3, 0.59, 0.04)
    cfg = tmp_path / "persona_config.yaml"
    cfg.write_text(yaml.dump({"layering": {**r3, "cancel_jitter": 0.6}}))   # the model's proposal
    tools = ToolRegistry(run_dir=run, persona_config_path=cfg)

    st = tools.effect_status("layering")
    assert st["status"] == "at_risk" and st["anchor_round"] == 2
    reverts = tools.enforce_effect_guard("layering", st)
    assert reverts == {"layer_spacing": [0.001, 0.0002], "n_layers": [2, 4]}
    written = yaml.safe_load(cfg.read_text())["layering"]
    assert written["cancel_jitter"] == 0.6 and written["layer_size_multiplier"] == 3.0   # evasion/helpful knobs kept


def test_effect_guard_undoes_new_effect_costing_change_only(tmp_path):
    import yaml
    from src.agentic.tools import ToolRegistry
    run = tmp_path / "run"
    r1 = {"layer_spacing": 0.0002, "n_layers": 4, "layer_size_multiplier": 2.0}
    r2 = {"layer_spacing": 0.0005, "n_layers": 4, "layer_size_multiplier": 2.0}
    _write_round(run, 0, r1, 1.0, None)
    _write_round(run, 1, r2, 0.6, 0.1)
    cfg = tmp_path / "persona_config.yaml"
    # the model undid the spacing change but newly shrank the layers
    cfg.write_text(yaml.dump({"layering": {**r2, "layer_spacing": 0.0003, "layer_size_multiplier": 1.0}}))
    tools = ToolRegistry(run_dir=run, persona_config_path=cfg)
    reverts = tools.enforce_effect_guard("layering", tools.effect_status("layering"))
    assert reverts == {"layer_size_multiplier": [1.0, 2.0]}
    assert yaml.safe_load(cfg.read_text())["layering"]["layer_spacing"] == 0.0003


def test_effect_surplus_status(tmp_path):
    from src.agentic.tools import ToolRegistry
    run = tmp_path / "run"
    _write_round(run, 0, {"stagger": 0.0}, 1.0, None)
    _write_round(run, 1, {"stagger": 0.9}, 1.8, 0.2)
    tools = ToolRegistry(run_dir=run, persona_config_path=tmp_path / "none.yaml")
    st = tools.effect_status("layering")
    assert st["status"] == "surplus"
    assert tools.enforce_effect_guard("layering", st) == {}


def test_cover_trading_inside_window_and_bounded():
    def run(rate, size):
        out = run_simulation(SimConfig(commodity="gold", n_ticks=1500, write_to_disk=False,
                                       enable_wash_trading=False, enable_pump_and_dump=False,
                                       spoof_start_tick=200, spoof_end_tick=700,
                                       spoof_cover_rate=rate, spoof_cover_size=size), 5)
        o = out.order_log
        cover = o[(o["event_type"] == "placed") & o["order_id"].str.endswith("v")]
        return out, cover
    out, cover = run(0.0, 0.0)
    assert cover.empty                                                   # off by default
    out, cover = run(0.4, 2.0)
    assert len(cover) > 50 and cover["tick"].between(200, 700).all()     # only inside the window
    assert cover["quantity"].max() < 50 * cover["quantity"].median()     # no depth feedback runaway
    ev = out.manipulation_events
    params = ev[ev["persona"] == "spoofing"].iloc[0]["parameters"]
    assert params["cover_rate"] == 0.4 and params["cover_size"] == 2.0

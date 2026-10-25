"""
tests/unit/test_agentic.py
==========================
Module 4 guardrails with a mocked LLM (no network, no API spend):
- propose_mutation bounds validation (FR-4.2)
- strategist: one re-prompt after a rejected proposal, then hand-authored fallback
  for the TARGET round
- mutation library indexing
- analyst tools return data or explicit errors, never invented numbers (FR-4.3)
- narrative grounding check rejects hallucinated numbers (FR-4.1)
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from src.agentic.adversarial_strategist import AdversarialStrategist, STRATEGIST_SYSTEM_PROMPT
from src.agentic.mutation_library import FALLBACK_MUTATIONS, get_fallback_mutation
from src.agentic.narrative_generator import NarrativeGenerator
from src.agentic.tools import ToolRegistry, validate_mutation


@pytest.fixture(autouse=True)
def _hermetic_llm(monkeypatch, tmp_path):
    """Strategist tests use a scripted client; make availability independent of the real .env."""
    import src.agentic.llm_client as lc
    monkeypatch.setenv("LLM_PROVIDER", "cerebras")
    monkeypatch.setenv("CEREBRAS_API_KEY", "csk-test-not-real")
    monkeypatch.setattr(lc, "_DISABLED", {})
    ledger = lc.CostLedger({"budget_usd": 1.0, "ledger_file": str(tmp_path / "spend.json")})
    ledger.path = tmp_path / "spend.json"
    monkeypatch.setattr(lc.CostLedger, "_inst", ledger)
    monkeypatch.setattr(lc.litellm, "completion", lambda **kw: pytest.fail("real LLM call attempted"))


@pytest.fixture
def registry(tmp_path) -> ToolRegistry:
    cfg = tmp_path / "persona_config.yaml"
    cfg.write_text(yaml.dump({"round_0": {"spoofing": {"enabled": True, "frequency": 0.4}}}))
    run = tmp_path / "run"
    (run / "round_0").mkdir(parents=True)
    (run / "round_0" / "metrics.json").write_text(json.dumps({
        "round": 0, "personas": {"spoofing": {"precision": 0.9, "recall": 0.8, "f1": 0.85}}, "n_flags": 1,
    }))
    return ToolRegistry(run_dir=run, persona_config_path=cfg)


def test_validate_mutation_rules():
    assert validate_mutation("spoofing", {"frequency": 0.3})[1] is None
    assert "outside" in validate_mutation("spoofing", {"frequency": 3.0})[1]
    assert "not a spoofing parameter" in validate_mutation("spoofing", {"n_layers": 3})[1]
    assert "integer" in validate_mutation("spoofing", {"cancellation_delay_ticks": 2.5})[1]
    assert "non-mutable" in validate_mutation("cornering", {"n_layers": 3})[1]
    assert validate_mutation("layering", {"n_layers": 3, "size_taper": 0.5})[1] is None
    assert validate_mutation("wash_trading", {})[1] is not None


def test_propose_mutation_rejects_without_writing(registry):
    before = registry.persona_config_path.read_text()
    res = registry.propose_mutation("spoofing", {"frequency": 5.0})
    assert "error" in res
    assert registry.persona_config_path.read_text() == before
    ok = registry.propose_mutation("spoofing", {"frequency": 0.2})
    assert ok["status"] == "success"
    assert yaml.safe_load(registry.persona_config_path.read_text())["spoofing"]["frequency"] == 0.2


def test_fallback_index_targets_round_and_clamps():
    assert get_fallback_mutation("spoofing", 1) == FALLBACK_MUTATIONS["spoofing"][0]
    assert get_fallback_mutation("spoofing", 2) == FALLBACK_MUTATIONS["spoofing"][1]
    assert get_fallback_mutation("spoofing", 9) == FALLBACK_MUTATIONS["spoofing"][-1]
    for persona, variants in FALLBACK_MUTATIONS.items():
        for v in variants:
            assert validate_mutation(persona, v["parameters"])[1] is None, (persona, v)


class _ScriptedLLM:
    """Returns one canned response per call."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0
        self.message_lengths = []
        self.messages = []

    def complete(self, system, messages, tools=None):
        self.calls += 1
        self.message_lengths.append(len(messages))
        self.messages.append(messages)
        if not self.responses:
            raise RuntimeError("quota exhausted")
        return self.responses.pop(0)


def _proposal(persona, params, call_id="c1"):
    return {"choices": [{"message": {"role": "assistant", "content": None, "tool_calls": [{
        "id": call_id, "type": "function",
        "function": {"name": "propose_mutation",
                     "arguments": json.dumps({"persona": persona, "parameters": params, "reasoning": "1. test"})},
    }]}}]}


def _strategist(registry, responses) -> AdversarialStrategist:
    s = AdversarialStrategist({"provider": "gemini", "max_tool_calls": 6}, tools=registry)
    s.client = _ScriptedLLM(responses)
    return s


def test_strategist_accepts_valid_proposal(registry):
    s = _strategist(registry, [_proposal("spoofing", {"frequency": 0.25})])
    out = s.run_strategist_round(round_num=0, personas=["spoofing"], target_round=1)
    assert out["status"] == "success" and out["source"] == {"spoofing": "agentic"}


def test_strategist_prompt_allows_broader_mutations():
    assert "change at least 4 knobs" in STRATEGIST_SYSTEM_PROMPT
    assert "up to 8" in STRATEGIST_SYSTEM_PROMPT
    assert "There is no 3-knob cap" in STRATEGIST_SYSTEM_PROMPT
    assert "change 1-3 knobs" not in STRATEGIST_SYSTEM_PROMPT


def test_strategist_compacts_messages_between_tool_turns(registry):
    s = _strategist(registry, [
        _proposal("spoofing", {"frequency": 0.25}),
        _proposal("layering", {"n_layers": 3}, "c2"),
    ])
    out = s.run_strategist_round(round_num=0, personas=["spoofing", "layering"], target_round=1)
    assert out["status"] == "success"
    assert s.client.calls == 2
    assert s.client.message_lengths == [1, 1]
    assert "Personas: spoofing." in s.client.messages[0][0]["content"]
    assert "Personas: layering." in s.client.messages[1][0]["content"]


def test_strategist_retries_truncated_response_for_same_persona(registry):
    truncated = {"choices": [{
        "message": {"role": "assistant", "content": None, "tool_calls": [{
            "id": "cut", "type": "function",
            "function": {"name": "propose_mutation", "arguments": '{"persona":"spoofing","parameters":{"fre'},
        }]},
        "finish_reason": "length",
    }]}
    s = _strategist(registry, [truncated, _proposal("spoofing", {"frequency": 0.25})])
    out = s.run_strategist_round(round_num=0, personas=["spoofing"], target_round=1)
    assert out["source"] == {"spoofing": "agentic"}
    assert s.client.calls == 2
    assert "truncated" in s.client.messages[1][0]["content"]
    assert "spoofing only" in s.client.messages[1][0]["content"]


def test_strategist_reprompts_once_then_falls_back_for_target_round(registry):
    bad = [_proposal("spoofing", {"frequency": 9.0}, "a"), _proposal("spoofing", {"frequency": 7.0}, "b")]
    s = _strategist(registry, bad)
    out = s.run_strategist_round(round_num=1, personas=["spoofing"], target_round=2)
    assert s.client.calls == 2, "exactly one re-prompt after the first rejection"
    assert s.client.message_lengths == [1, 1]
    assert s.client.messages[0][0]["content"] != s.client.messages[1][0]["content"]
    assert out["source"] == {"spoofing": "hand_authored"}
    written = yaml.safe_load(registry.persona_config_path.read_text())["spoofing"]
    expected = FALLBACK_MUTATIONS["spoofing"][1]["parameters"]       # round 2 → second variant
    assert all(written[k] == v for k, v in expected.items())


def test_strategist_reprompts_when_proposal_or_alternative_is_too_similar(registry, monkeypatch):
    import src.agentic.adversarial_strategist as strategist_module

    monkeypatch.setattr(strategist_module, "NOVELTY_ENFORCEMENT_AFTER_ROUND", 0)
    s = _strategist(registry, [
        _proposal("spoofing", {"frequency": 0.25}, "stale"),
        _proposal("spoofing", {"cancellation_delay_ticks": 12}, "fresh"),
    ])
    monkeypatch.setattr(s, "_recent_mutations", lambda *_: [
        {"round": 1, "knobs": {"frequency", "size_jitter"},
         "frozen_coverage": 0.9, "effect_ratio": 0.8, "score": 0.1}
    ])
    monkeypatch.setattr(s, "_probed_history", lambda *_: [])

    before_config = yaml.safe_load(registry.persona_config_path.read_text())
    before = before_config["round_0"]["spoofing"]
    out = s.run_strategist_round(round_num=1, personas=["spoofing"], target_round=2)

    assert out["source"] == {"spoofing": "agentic"}
    assert s.client.calls == 2
    assert "rejected before being applied" in s.client.messages[1][0]["content"]
    assert "round 1 already tried" in s.client.messages[1][0]["content"]
    assert "genuinely different tactic" in s.client.messages[1][0]["content"]
    updated = yaml.safe_load(registry.persona_config_path.read_text())["spoofing"]
    assert updated["frequency"] == before["frequency"]
    assert updated["cancellation_delay_ticks"] == 12


def test_strategist_llm_failure_uses_fallback(registry):
    s = _strategist(registry, [])
    out = s.run_strategist_round(round_num=0, personas=["spoofing"], target_round=1)
    assert out["status"] == "fallback" and out["source"]["spoofing"] == "hand_authored"


def test_round3_llm_failure_uses_novel_code_fallback(registry, monkeypatch):
    import src.agentic.adversarial_strategist as strategist_module

    monkeypatch.setattr(strategist_module, "llm_unavailable_reason", lambda _config: None)

    class FailedLLM:
        def complete(self, **_kwargs):
            raise RuntimeError("simulated TPM limit")

    s = AdversarialStrategist({"max_tool_calls": 1}, tools=registry)
    s.client = FailedLLM()
    out = s.run_strategist_round(round_num=0, personas=["spoofing"], target_round=3)
    assert out["status"] == "fallback"
    assert out["source"]["spoofing"] == "code_exploration"
    assert registry.candidates["spoofing"]


def test_tools_report_missing_data_instead_of_inventing(registry):
    assert "error" in registry.get_round_metrics(5)
    assert registry.get_round_metrics(0)["spoofing"]["f1"] == 0.85
    assert "error" in registry.query_flags(round_num=0)
    assert "error" in registry.get_trader_history("x", 0)
    assert registry.list_rounds() == {"rounds": [{"round_num": 0, "n_flags": 1}]}


def test_query_flags_reads_flags_parquet(registry, tmp_path):
    flags = pd.DataFrame([{
        "commodity": "gold", "window_start": 0, "trader_id": "spoofer_0", "window_end": 179,
        "final_flag": True, "final_score": 0.93, "iso_score": 0.5, "predicted_persona": "spoofing",
        "triggered_by": ["spoofing_classifier"], "top_shap_features": {"spoof_signature": 1.2},
        "persona_scores": {"spoofing": 0.93, "wash_trading": 0.01, "pump_and_dump": 0.02},
    }])
    flags.to_parquet(registry.run_dir / "round_0" / "flags.parquet")
    out = registry.query_flags(round_num=0, persona="spoofing")
    assert out["total_matching_flags"] == 1
    assert out["flags"][0]["top_shap_features"] == {"spoof_signature": 1.2}


def test_narrative_grounding_rejects_hallucinated_numbers():
    gen = NarrativeGenerator.__new__(NarrativeGenerator)
    data = {"flag": {"trader_id": "spoofer_0", "detector_score": 0.93}, "shap": {"spoof_signature": 1.21}}
    assert gen._validate_grounding("Score 0.93 driven by spoof_signature 1.21.", data)
    assert not gen._validate_grounding("Score 0.93 with a 340% cancel spike.", data)


# --- Shape of LLM candidates (min_llm_knobs / min_knob_move / llm_knob_coverage) ---

_SHAPED = {"provider": "gemini", "max_tool_calls": 6, "min_llm_knobs": 4, "min_knob_move": 0.1, "llm_knob_coverage": 0.6}
_WIDE_A = {"order_size_multiplier": 9.0, "frequency": 0.8, "cancel_jitter": 0.6, "n_split_orders": 3}
_WIDE_B = {"size_jitter": 0.6, "n_accounts": 3, "cover_rate": 0.3, "flip_after_cancel": 0.5}
_WIDE_C = {"busy_period_bias": 0.7, "cover_size": 2.0, "reprice_to_touch": 1, "cancellation_delay_ticks": 20}


def _shaped_proposal(params, alternatives, call_id="c1"):
    return {"choices": [{"message": {"role": "assistant", "content": None, "tool_calls": [{
        "id": call_id, "type": "function",
        "function": {"name": "propose_mutation",
                     "arguments": json.dumps({"persona": "spoofing", "parameters": params,
                                              "alternatives": alternatives, "reasoning": "1. test"})},
    }]}}]}


def test_shape_rules_accept_broad_bold_proposal(registry):
    s = AdversarialStrategist(_SHAPED, tools=registry)
    s.client = _ScriptedLLM([_shaped_proposal(_WIDE_A, [_WIDE_B, _WIDE_C])])
    out = s.run_strategist_round(round_num=0, personas=["spoofing"], target_round=1)
    assert out["source"] == {"spoofing": "agentic"} and s.client.calls == 1
    assert len(registry.candidates["spoofing"]) == 3


def test_shape_rules_reject_narrow_proposal_then_accept_retry(registry):
    s = AdversarialStrategist(_SHAPED, tools=registry)
    s.client = _ScriptedLLM([
        _shaped_proposal({"frequency": 0.8}, []),                       # 1 knob, no alternatives: rejected
        _shaped_proposal(_WIDE_A, [{"frequency": 0.9}], "c2"),          # retry: primary checked, thin alt dropped
    ])
    out = s.run_strategist_round(round_num=0, personas=["spoofing"], target_round=1)
    assert out["source"] == {"spoofing": "agentic"} and s.client.calls == 2
    assert "proposal shape rejected" in s.client.messages[1][0]["content"]
    assert len(registry.candidates["spoofing"]) == 1


def test_shape_rules_ignore_tiny_moves(registry):
    s = AdversarialStrategist(_SHAPED, tools=registry)
    current = s._current_params("spoofing", 0)
    tiny = {"frequency": float(current.get("frequency", 0.4)) + 0.01}   # 1% of the 0.05-1.0 range
    assert s._real_changes("spoofing", current, {**tiny, "order_size_multiplier": 9.0}) == {"order_size_multiplier": 9.0}


def test_shape_rules_drop_thin_alternative_instead_of_rejecting(registry):
    s = AdversarialStrategist(_SHAPED, tools=registry)
    s.client = _ScriptedLLM([_shaped_proposal(_WIDE_A, [{**_WIDE_B, **_WIDE_C}, {"frequency": 0.9}])])
    out = s.run_strategist_round(round_num=0, personas=["spoofing"], target_round=1)
    assert out["source"] == {"spoofing": "agentic"} and s.client.calls == 1
    assert len(registry.candidates["spoofing"]) == 2   # primary + the broad alternative; the 1-knob one dropped

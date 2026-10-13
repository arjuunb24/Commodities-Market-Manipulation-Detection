"""
tests/unit/test_llm_client.py
=============================
Cost / safety guards of the LLM client. litellm.completion is mocked — these
tests never contact a provider or spend credit.
"""

from __future__ import annotations

import json

import pytest

import src.agentic.llm_client as lc


class _Resp:
    def __init__(self, prompt=1000, completion=200):
        self._d = {"choices": [{"message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
                   "usage": {"prompt_tokens": prompt, "completion_tokens": completion}}

    def model_dump_json(self):
        return json.dumps(self._d)


@pytest.fixture
def cerebras(monkeypatch, tmp_path):
    monkeypatch.setenv("LLM_PROVIDER", "cerebras")
    monkeypatch.setenv("CEREBRAS_API_KEY", "csk-test-key-1234567890")
    monkeypatch.setattr(lc, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(lc, "_DISABLED", {})
    ledger = lc.CostLedger({"budget_usd": 0.01, "ledger_file": str(tmp_path / "spend.json"),
                            "min_seconds_between_calls": 0, "max_calls_per_process": 5})
    ledger.path = tmp_path / "spend.json"
    monkeypatch.setattr(lc.CostLedger, "_inst", ledger)
    monkeypatch.setattr(lc.time, "sleep", lambda s: None)
    calls = []
    return ledger, calls


def _client(tmp_path, **cfg):
    return lc.LLMClient({"max_tokens": {"cerebras": 300}, "max_retries": 2, **cfg}, run_dir=tmp_path / "run")


def test_switch_routes_to_cerebras_only(cerebras, monkeypatch, tmp_path):
    _, calls = cerebras
    monkeypatch.setattr(lc.litellm, "completion", lambda **kw: calls.append(kw) or _Resp())
    c = _client(tmp_path)
    assert c.full_model_str == "cerebras/gpt-oss-120b"
    c.complete("sys", [{"role": "user", "content": "hi"}])
    assert calls[0]["api_key"] == "csk-test-key-1234567890"
    assert calls[0]["max_retries"] == 0, "hidden SDK retries must be disabled"
    assert calls[0]["max_tokens"] == 300


def test_cost_recorded_and_key_not_logged(cerebras, monkeypatch, tmp_path):
    ledger, _ = cerebras
    monkeypatch.setattr(lc.litellm, "completion", lambda **kw: _Resp(1000, 200))
    _client(tmp_path).complete("sys", [{"role": "user", "content": "hi"}])
    assert ledger.state["spent_usd"] == pytest.approx(1000 * 0.35e-6 + 200 * 0.75e-6)
    log = (tmp_path / "run" / "llm_calls.jsonl").read_text()
    assert "csk-test-key" not in log


def test_cache_prevents_repeat_billing(cerebras, monkeypatch, tmp_path):
    _, calls = cerebras
    monkeypatch.setattr(lc.litellm, "completion", lambda **kw: calls.append(1) or _Resp())
    c = _client(tmp_path)
    c.complete("sys", [{"role": "user", "content": "same"}])
    c.complete("sys", [{"role": "user", "content": "same"}])
    assert len(calls) == 1


def test_budget_refuses_before_sending(cerebras, monkeypatch, tmp_path):
    ledger, calls = cerebras
    ledger.state["spent_usd"] = 0.0099
    monkeypatch.setattr(lc.litellm, "completion", lambda **kw: calls.append(1) or _Resp())
    with pytest.raises(lc.BudgetExceededError):
        _client(tmp_path).complete("sys", [{"role": "user", "content": "x" * 4000}])
    assert calls == []
    assert lc.llm_unavailable_reason({}) is None or "budget" in lc.llm_unavailable_reason({})


def test_fatal_error_not_retried_and_disables_provider(cerebras, monkeypatch, tmp_path):
    _, calls = cerebras

    def boom(**kw):
        calls.append(1)
        raise Exception("402 Payment Required: insufficient credit")

    monkeypatch.setattr(lc.litellm, "completion", boom)
    c = _client(tmp_path)
    with pytest.raises(lc.LLMUnavailableError):
        c.complete("sys", [{"role": "user", "content": "a"}])
    with pytest.raises(lc.LLMUnavailableError):
        c.complete("sys", [{"role": "user", "content": "b"}])
    assert len(calls) == 1, "after a fatal error no further requests are sent"
    assert "disabled" in lc.llm_unavailable_reason({})


def test_rate_limit_retries_are_bounded(cerebras, monkeypatch, tmp_path):
    _, calls = cerebras

    def limited(**kw):
        calls.append(1)
        raise Exception("429 Too Many Requests")

    monkeypatch.setattr(lc.litellm, "completion", limited)
    with pytest.raises(lc.LLMUnavailableError):
        _client(tmp_path).complete("sys", [{"role": "user", "content": "a"}])
    assert len(calls) == 3  # 1 + max_retries(2), never more


def test_gemini_switch_never_reads_cerebras_key(monkeypatch, tmp_path):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setattr(lc, "_GLOBAL_GEMINI_POOL", None)
    for k in ("GEMINI_API_KEY", "GEMINI_STRATEGIST_API_KEY", "GEMINI_GENERAL_API_KEYS"):
        monkeypatch.setenv(k, "")
    c = lc.LLMClient({}, run_dir=None)
    assert c.full_model_str.startswith("gemini/")
    with pytest.raises(lc.LLMUnavailableError):
        c.complete("sys", [{"role": "user", "content": "a"}])

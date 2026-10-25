"""
src/agentic/llm_client.py
=========================
Single LLM entry point for Module 4 (narratives, strategist, analyst chat).

Provider switch
---------------
One switch selects the provider for EVERY LLM call in the project:
    env LLM_PROVIDER=cerebras|gemini        (wins if set)
    configs/llm_config.yaml: active_provider (default)
  * cerebras → only CEREBRAS_API_KEY is used (gpt-oss-120b). Gemini keys are never read.
  * gemini   → the Gemini multi-key pool with 429 failover (previous behaviour).
Component blocks in llm_config.yaml (narrative_generator, adversarial_strategist,
analyst_assistant) only carry per-component limits; model/temperature come from
the active provider's profile.

Cost and safety guards (paid Cerebras credit)
---------------------------------------------
  * Budget ledger persisted in data/llm_cache/cerebras_spend.json. Before each
    call the WORST-CASE cost (estimated input + max_tokens output) is checked
    against the remaining budget; if it does not fit, no request is sent.
  * Hard per-process call cap and a minimum interval between calls (stays under
    per-minute rate limits instead of hitting 429s).
  * Identical requests are served from the response cache, never re-billed.
  * Retries are bounded: auth / billing / bad-request errors are never retried and
    disable the provider for the rest of the process (no error loops); 429 and
    5xx/timeouts get at most `max_retries` attempts with backoff. The OpenAI SDK's
    own hidden retries are switched off so attempts are never multiplied.
  * API keys are stripped from every logged request.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import socket
import threading
import time
from collections import deque
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

import litellm
import yaml
from dotenv import load_dotenv

load_dotenv()
litellm.drop_params = True          # never fail on a param a provider does not support
litellm.suppress_debug_info = True

# Prefer IPv4 on Windows to avoid IPv6 connection-timeout stalls.
try:
    _orig_getaddrinfo = socket.getaddrinfo

    def _ipv4_first_getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
        res = _orig_getaddrinfo(host, port, family, type, proto, flags)
        return sorted(res, key=lambda x: 0 if x[0] == socket.AF_INET else 1)

    socket.getaddrinfo = _ipv4_first_getaddrinfo
except Exception:
    pass

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
LLM_CONFIG_PATH = REPO_ROOT / "configs" / "llm_config.yaml"
CACHE_DIR = REPO_ROOT / "data" / "llm_cache"


# =============================================================================
# Errors
# =============================================================================

class LLMUnavailableError(RuntimeError):
    """No usable provider/key, or the provider was disabled after a fatal error (SRS FR-4.4)."""


class BudgetExceededError(LLMUnavailableError):
    """The request could exceed the remaining Cerebras budget; it was not sent."""


# =============================================================================
# Settings / provider switch
# =============================================================================

@lru_cache(maxsize=1)
def load_llm_settings() -> dict:
    if LLM_CONFIG_PATH.exists():
        return yaml.safe_load(LLM_CONFIG_PATH.read_text()) or {}
    return {}


def active_provider() -> str:
    env = (os.getenv("LLM_PROVIDER") or "").strip().lower()
    if env in ("cerebras", "gemini"):
        return env
    return str(load_llm_settings().get("active_provider", "cerebras")).lower()


def _usable(v: Optional[str]) -> bool:
    return bool(v) and not v.strip().lower().startswith("your-")


def llm_unavailable_reason(config: dict[str, Any] | None = None) -> Optional[str]:
    """
    None if the ACTIVE provider can serve calls, else a plain-language reason.
    Lets callers fail fast at start-up instead of discovering problems mid-run.
    """
    provider = active_provider()
    if provider in _DISABLED:
        return f"{provider} calls were disabled after an error: {_DISABLED[provider]}"
    if provider == "cerebras":
        if not _usable(os.getenv("CEREBRAS_API_KEY")):
            return "No Cerebras API key configured (set CEREBRAS_API_KEY in .env)."
        ledger = CostLedger.instance()
        if ledger.remaining() <= 0.0:
            return f"Cerebras budget of ${ledger.budget:.2f} is used up (see {ledger.path.name})."
        return None
    keys = [os.getenv("GEMINI_API_KEY"), os.getenv("GEMINI_STRATEGIST_API_KEY")]
    keys += (os.getenv("GEMINI_GENERAL_API_KEYS") or "").split(",")
    if any(_usable(k) for k in keys):
        return None
    if os.getenv("DEMO_MODE", "false").lower() == "true" and any(CACHE_DIR.glob("*.json")):
        return None
    return "No Gemini API key configured (set GEMINI_API_KEY in .env)."


# Providers disabled for the rest of the process after a non-retryable error.
_DISABLED: dict[str, str] = {}


# =============================================================================
# Cerebras budget ledger
# =============================================================================

class CostLedger:
    """Persistent spend tracker with worst-case pre-checks (thread-safe, process-wide)."""

    _inst: Optional["CostLedger"] = None
    _lock = threading.Lock()

    def __init__(self, cfg: dict) -> None:
        self.budget = float(cfg.get("budget_usd", 4.5))
        self.in_price = float(cfg.get("input_usd_per_million", 0.35)) / 1e6
        self.out_price = float(cfg.get("output_usd_per_million", 0.75)) / 1e6
        self.max_calls = int(cfg.get("max_calls_per_process", 80))
        self.min_interval = float(cfg.get("min_seconds_between_calls", 2.0))
        self.tokens_per_minute = int(cfg.get("tokens_per_minute", 20_000))
        self.path = REPO_ROOT / cfg.get("ledger_file", "data/llm_cache/cerebras_spend.json")
        self.calls_this_process = 0
        self._last_call = 0.0
        self._token_reservations: deque[tuple[float, int]] = deque()
        self.state = {"spent_usd": 0.0, "calls": 0, "input_tokens": 0, "output_tokens": 0}
        if self.path.exists():
            try:
                self.state.update(json.loads(self.path.read_text()))
            except Exception:
                logger.warning("Could not read %s; starting a new ledger.", self.path)

    @classmethod
    def instance(cls) -> "CostLedger":
        with cls._lock:
            if cls._inst is None:
                cls._inst = cls(load_llm_settings().get("cerebras_budget", {}))
            return cls._inst

    def remaining(self) -> float:
        return self.budget - float(self.state["spent_usd"])

    def worst_case(self, input_tokens: int, max_tokens: int) -> float:
        return input_tokens * self.in_price + max_tokens * self.out_price

    def reserve(self, input_tokens: int, max_tokens: int) -> None:
        """Reserve worst-case spend and TPM capacity before sending a request."""
        with self._lock:
            if self.calls_this_process >= self.max_calls:
                raise BudgetExceededError(
                    f"Per-process Cerebras call cap reached ({self.max_calls}); raise "
                    "cerebras_budget.max_calls_per_process in llm_config.yaml if intended.")
            request_tokens = input_tokens + max_tokens
            if self.tokens_per_minute > 0 and request_tokens > self.tokens_per_minute:
                raise BudgetExceededError(
                    f"Request reserves {request_tokens} tokens, exceeding the configured Cerebras "
                    f"TPM budget of {self.tokens_per_minute}; reduce prompt or max_tokens.")
            cost = self.worst_case(input_tokens, max_tokens)
            if cost > self.remaining():
                raise BudgetExceededError(
                    f"Request worst-case ${cost:.4f} exceeds remaining Cerebras budget ${self.remaining():.4f}.")
            if self.tokens_per_minute > 0:
                while True:
                    now = time.monotonic()
                    while self._token_reservations and now - self._token_reservations[0][0] >= 60:
                        self._token_reservations.popleft()
                    used = sum(tokens for _, tokens in self._token_reservations)
                    if used + request_tokens <= self.tokens_per_minute:
                        break
                    wait = 60 - (now - self._token_reservations[0][0]) + 0.1
                    logger.info(
                        "[Cerebras] TPM budget reserved (%d/%d tokens); waiting %.1fs before request.",
                        used, self.tokens_per_minute, wait,
                    )
                    time.sleep(wait)
            wait = self.min_interval - (time.time() - self._last_call)
            if wait > 0:
                time.sleep(wait)
            self._last_call = time.time()
            self.calls_this_process += 1
            if self.tokens_per_minute > 0:
                self._token_reservations.append((time.monotonic(), request_tokens))

    def record(self, prompt_tokens: int, completion_tokens: int) -> float:
        cost = prompt_tokens * self.in_price + completion_tokens * self.out_price
        with self._lock:
            s = self.state
            s["spent_usd"] = round(float(s["spent_usd"]) + cost, 6)
            s["calls"] = int(s["calls"]) + 1
            s["input_tokens"] = int(s["input_tokens"]) + prompt_tokens
            s["output_tokens"] = int(s["output_tokens"]) + completion_tokens
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(s, indent=2))
            os.replace(tmp, self.path)
        logger.info("[Cerebras] call cost $%.5f — total spent $%.4f of $%.2f",
                    cost, s["spent_usd"], self.budget)
        return cost


# =============================================================================
# Gemini key pool (only used when the switch is set to gemini)
# =============================================================================

class GeminiKeyPool:
    """Round-robin Gemini keys with cooldown on 429 / ResourceExhausted."""

    def __init__(self) -> None:
        self._keys: list[str] = self._discover_keys()
        self._current_idx = 0
        self._cooldowns: dict[str, float] = {}

    def _discover_keys(self) -> list[str]:
        raw = [os.getenv("GEMINI_API_KEY"), os.getenv("GEMINI_STRATEGIST_API_KEY")]
        raw += (os.getenv("GEMINI_GENERAL_API_KEYS") or "").split(",")
        unique: list[str] = []
        for k in raw:
            k = (k or "").strip()
            if _usable(k) and k not in unique:
                unique.append(k)
        if unique:
            logger.info(f"[GeminiKeyPool] Loaded {len(unique)} unique Gemini API keys.")
        else:
            logger.warning("[GeminiKeyPool] No Gemini API keys found in environment variables.")
        return unique

    @property
    def total_keys(self) -> int:
        return len(self._keys)

    def get_next_key(self) -> Optional[str]:
        if not self._keys:
            return None
        now = time.time()
        for i in range(len(self._keys)):
            idx = (self._current_idx + i) % len(self._keys)
            key = self._keys[idx]
            if now >= self._cooldowns.get(key, 0.0):
                self._current_idx = (idx + 1) % len(self._keys)
                return key
        return None  # every key is cooling down — caller stops instead of looping

    def mark_rate_limited(self, key: str, cooldown_seconds: float = 60.0) -> None:
        self._cooldowns[key] = time.time() + cooldown_seconds
        logger.warning(f"[GeminiKeyPool] Key ...{key[-4:]} on cooldown for {cooldown_seconds:.0f}s.")


_GLOBAL_GEMINI_POOL: Optional[GeminiKeyPool] = None


def get_gemini_pool() -> GeminiKeyPool:
    global _GLOBAL_GEMINI_POOL
    if _GLOBAL_GEMINI_POOL is None:
        _GLOBAL_GEMINI_POOL = GeminiKeyPool()
    return _GLOBAL_GEMINI_POOL


# =============================================================================
# Error classification
# =============================================================================

_FATAL_PATTERNS = ("401", "403", "402", "invalid api key", "authentication", "unauthorized",
                   "permission", "insufficient", "billing", "payment", "credit", "exceeded your")
_RATE_PATTERNS = (
    "429", "rate limit", "rate_limit", "too many requests", "resource_exhausted", "quota",
    "ratelimiterror", "tokens per minute", "token per minute", "too many tokens processed", "tpm limit",
)
_TRANSIENT_PATTERNS = ("500", "502", "503", "504", "timeout", "timed out", "unavailable",
                       "connection", "overloaded", "high demand")


def _classify(exc: Exception) -> str:
    msg = f"{type(exc).__name__} {exc}".lower()
    if any(p in msg for p in _FATAL_PATTERNS) and "rate" not in msg:
        return "fatal"
    if any(p in msg for p in _RATE_PATTERNS):
        return "rate"
    if "400" in msg or "badrequest" in msg or "invalid_request" in msg or "context length" in msg:
        return "bad_request"
    if any(p in msg for p in _TRANSIENT_PATTERNS):
        return "transient"
    return "bad_request"  # unknown → do not retry blindly


def _retry_after(exc: Exception, default: float) -> float:
    m = re.search(r"retry[- ]after[^0-9]*([0-9.]+)", str(exc).lower())
    return min(float(m.group(1)), 60.0) if m else default


def _sanitize_messages(messages: list[dict]) -> list[dict]:
    """
    Keep only standard chat fields. Assistant messages returned by litellm carry
    extras (reasoning_content, provider_specific_fields, ...) that Cerebras
    rejects when they are sent back on the next turn of a tool-calling loop.
    """
    clean: list[dict] = []
    for m in messages:
        out: dict[str, Any] = {"role": m.get("role"), "content": m.get("content")}
        if m.get("tool_calls"):
            out["tool_calls"] = [
                {"id": tc.get("id"), "type": tc.get("type") or "function",
                 "function": {"name": tc["function"]["name"], "arguments": tc["function"].get("arguments") or "{}"}}
                for tc in m["tool_calls"]
            ]
            if out["content"] is None:
                out["content"] = ""
        if m.get("role") == "tool":
            out["tool_call_id"] = m.get("tool_call_id")
            if m.get("name"):
                out["name"] = m["name"]
        clean.append(out)
    return clean


def _estimate_tokens(*parts: Any) -> int:
    text = "".join(p if isinstance(p, str) else json.dumps(p, default=str) for p in parts if p)
    return int(len(text) / 3.5) + 20


# =============================================================================
# LLM client
# =============================================================================

class LLMClient:
    def __init__(self, config: dict[str, Any], run_dir: Optional[Path] = None) -> None:
        """
        Args:
            config: one component block of llm_config.yaml (e.g. ['narrative_generator']).
                    Model/temperature come from the ACTIVE provider's profile; a value
                    may be a {gemini: x, cerebras: y} mapping to differ per provider.
            run_dir: run directory for the llm_calls.jsonl audit log.
        """
        settings = load_llm_settings()
        self.provider = active_provider()
        profile = settings.get("providers", {}).get(self.provider, {})

        def pick(key: str, default: Any) -> Any:
            v = config.get(key, profile.get(key, default))
            return v.get(self.provider, default) if isinstance(v, dict) else v

        self.model_name = pick("model", profile.get("model", "gpt-oss-120b"))
        prefix = f"{self.provider}/"
        self.full_model_str = self.model_name if self.model_name.startswith(prefix) else prefix + self.model_name
        self.temperature = pick("temperature", 0.2)
        self.max_tokens = int(pick("max_tokens", 1024))
        self.max_retries = int(pick("max_retries", 2))
        self.timeout = float(pick("timeout_seconds", 60))
        self.reasoning_effort = pick("reasoning_effort", None)

        self.run_dir = Path(run_dir) if run_dir else None
        self.log_file = self.run_dir / "llm_calls.jsonl" if self.run_dir else None
        self.demo_mode = os.getenv("DEMO_MODE", "false").lower() == "true"
        self.use_cache = bool(profile.get("cache_responses", True)) or self.demo_mode
        self.cache_dir = CACHE_DIR
        self.key_pool = get_gemini_pool() if self.provider == "gemini" else None

    # ------------------------------------------------------------------ cache
    def _hash_input(self, system: str, messages: list[dict], tools: Optional[list[dict]]) -> str:
        payload = {"model": self.full_model_str, "max_tokens": self.max_tokens,
                   "system": system, "messages": messages, "tools": tools}
        return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()

    def _read_cache(self, key: str) -> Optional[dict]:
        p = self.cache_dir / f"{key}.json"
        if p.exists():
            try:
                return json.loads(p.read_text())
            except Exception:
                return None
        return None

    def _write_cache(self, key: str, data: dict) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        (self.cache_dir / f"{key}.json").write_text(json.dumps(data))

    # ------------------------------------------------------------------ audit
    def _log_call(self, payload: dict, response: dict, latency_ms: float, cost: float | None) -> None:
        if not self.log_file:
            return
        safe = {k: v for k, v in payload.items() if k not in ("api_key", "extra_headers")}
        entry = {
            "timestamp": time.time(),
            "provider": self.provider,
            "model": self.full_model_str,
            "latency_ms": round(latency_ms, 1),
            "usage": response.get("usage"),
            "cost_usd": cost,
            "request": safe,
            "response": response,
        }
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.log_file, "a") as f:
            f.write(json.dumps(entry, default=str) + "\n")

    # ------------------------------------------------------------------ call
    def complete(self, system: str, messages: list[dict], tools: Optional[list[dict]] = None) -> dict:
        """
        One chat completion. Returns the OpenAI-style response dict.

        Raises LLMUnavailableError / BudgetExceededError instead of looping or
        overspending; callers fall back (SRS FR-4.4).
        """
        if self.provider in _DISABLED:
            raise LLMUnavailableError(f"{self.provider} disabled for this process: {_DISABLED[self.provider]}")

        messages = _sanitize_messages(messages)
        cache_key = self._hash_input(system, messages, tools)
        if self.use_cache:
            cached = self._read_cache(cache_key)
            if cached:
                logger.info("[LLMClient] cache hit — no API call")
                return cached
            if self.demo_mode:
                logger.warning("[LLMClient] cache miss in DEMO_MODE; making a live call.")

        call_kwargs: dict[str, Any] = {
            "model": self.full_model_str,
            "messages": [{"role": "system", "content": system}] + messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "timeout": self.timeout,
            "max_retries": 0,     # no hidden SDK retries — attempts are counted here only
            "num_retries": 0,
        }
        if tools:
            call_kwargs["tools"] = tools
        if self.reasoning_effort:
            call_kwargs["reasoning_effort"] = self.reasoning_effort

        if self.provider == "cerebras":
            response = self._complete_cerebras(call_kwargs)
        else:
            response = self._complete_gemini(call_kwargs)

        if self.use_cache and not any(
            choice.get("finish_reason") == "length"
            for choice in response.get("choices", [])
        ):
            self._write_cache(cache_key, response)
        return response

    def _complete_cerebras(self, call_kwargs: dict) -> dict:
        key = os.getenv("CEREBRAS_API_KEY", "")
        if not _usable(key):
            raise LLMUnavailableError("No Cerebras API key configured (set CEREBRAS_API_KEY in .env).")
        call_kwargs["api_key"] = key
        ledger = CostLedger.instance()
        est_in = _estimate_tokens(call_kwargs["messages"], call_kwargs.get("tools"))
        attempts = 1 + max(0, self.max_retries)
        last: Exception | None = None
        for attempt in range(1, attempts + 1):
            ledger.reserve(est_in, self.max_tokens)          # raises before any overspend
            start = time.time()
            try:
                raw = litellm.completion(**call_kwargs)
                response = json.loads(raw.model_dump_json())
            except Exception as exc:
                last = exc
                kind = _classify(exc)
                logger.warning(f"[Cerebras] attempt {attempt}/{attempts} failed ({kind}): {str(exc)[:200]}")
                if kind == "fatal":
                    _DISABLED["cerebras"] = str(exc)[:200]
                    raise LLMUnavailableError(f"Cerebras rejected the request: {str(exc)[:200]}") from exc
                if kind == "bad_request" or attempt == attempts:
                    raise LLMUnavailableError(f"Cerebras call failed: {str(exc)[:200]}") from exc
                time.sleep(_retry_after(exc, 60.0) if kind == "rate" else 2.0 * attempt)
                continue
            usage = response.get("usage") or {}
            cost = ledger.record(int(usage.get("prompt_tokens") or est_in),
                                 int(usage.get("completion_tokens") or self.max_tokens))
            self._log_call(call_kwargs, response, (time.time() - start) * 1000, cost)
            return response
        raise LLMUnavailableError(f"Cerebras call failed: {last}")

    def _complete_gemini(self, call_kwargs: dict) -> dict:
        pool = self.key_pool
        if pool is None or pool.total_keys == 0:
            raise LLMUnavailableError("No Gemini API key configured (set GEMINI_API_KEY in .env).")
        attempts = max(1 + self.max_retries, pool.total_keys)
        last: Exception | None = None
        for attempt in range(1, attempts + 1):
            key = pool.get_next_key()
            if key is None:
                raise LLMUnavailableError("All Gemini keys are rate-limited; try again later.")
            call_kwargs["api_key"] = key
            start = time.time()
            try:
                raw = litellm.completion(**call_kwargs)
                response = json.loads(raw.model_dump_json())
                self._log_call(call_kwargs, response, (time.time() - start) * 1000, None)
                return response
            except Exception as exc:
                last = exc
                kind = _classify(exc)
                logger.warning(f"[Gemini] attempt {attempt}/{attempts} failed ({kind}): {str(exc)[:200]}")
                if kind == "rate":
                    pool.mark_rate_limited(key, 3600.0 if "quota" in str(exc).lower() else 60.0)
                elif kind == "transient":
                    pool.mark_rate_limited(key, 10.0)
                    time.sleep(2.0)
                else:
                    raise LLMUnavailableError(f"Gemini call failed: {str(exc)[:200]}") from exc
        raise LLMUnavailableError(f"Gemini call failed after {attempts} attempts: {last}")

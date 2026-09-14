"""
src/agentic/llm_client.py
=========================
A unified wrapper around LiteLLM for routing between Gemini and Cerebras.
Handles retries, logging, and DEMO_MODE caching.
"""

import json
import os
import hashlib
import time
import logging
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv
import litellm
from tenacity import Retrying, stop_after_attempt, wait_exponential

# Load environment variables (API keys, DEMO_MODE)
load_dotenv()

logger = logging.getLogger(__name__)

class LLMClient:
    def __init__(self, config: dict[str, Any], run_dir: Optional[Path] = None) -> None:
        """
        Args:
            config: A specific sub-config from llm_config.yaml (e.g., config['narrative_generator'])
            run_dir: The directory of the current simulation run, for logging.
        """
        self.provider = config.get("provider", "gemini")
        self.model_name = config.get("model", "gemini-2.0-flash")
        
        # litellm expects the model string in the format "provider/model" if it's not OpenAI
        # For Gemini: "gemini/gemini-2.0-flash"
        # For Cerebras: "cerebras/llama-3.3-70b" (or similar supported prefix)
        if self.provider == "gemini" and not self.model_name.startswith("gemini/"):
            self.full_model_str = f"gemini/{self.model_name}"
        elif self.provider == "cerebras" and not self.model_name.startswith("cerebras/"):
            self.full_model_str = f"cerebras/{self.model_name}"
        else:
            self.full_model_str = self.model_name
            
        self.temperature = config.get("temperature", 0.0)
        self.max_tokens = config.get("max_tokens", 1024)
        self.max_retries = config.get("max_retries", 3)
        self.timeout = config.get("timeout_seconds", 30)
        
        self.run_dir = run_dir
        self.log_file = self.run_dir / "llm_calls.jsonl" if self.run_dir else None
        
        self.demo_mode = os.getenv("DEMO_MODE", "false").lower() == "true"
        self.cache_dir = Path("data/llm_cache")
        if self.demo_mode:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            
    def _hash_input(self, system: str, messages: list[dict], tools: Optional[list[dict]]) -> str:
        payload = {"system": system, "messages": messages, "tools": tools}
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()

    def _read_cache(self, cache_key: str) -> Optional[dict]:
        cache_path = self.cache_dir / f"{cache_key}.json"
        if cache_path.exists():
            with open(cache_path, "r") as f:
                return json.load(f)
        return None

    def _write_cache(self, cache_key: str, response_data: dict) -> None:
        cache_path = self.cache_dir / f"{cache_key}.json"
        with open(cache_path, "w") as f:
            json.dump(response_data, f, indent=2)

    def _log_call(self, payload: dict, response: dict, latency_ms: float) -> None:
        if not self.log_file:
            return
            
        log_entry = {
            "timestamp": time.time(),
            "model": self.full_model_str,
            "latency_ms": latency_ms,
            "request": payload,
            "response": response
        }
        
        with open(self.log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

    def complete(
        self, 
        system: str, 
        messages: list[dict], 
        tools: Optional[list[dict]] = None
    ) -> dict:
        """
        Executes a completion request against the LLM, handling retries and caching.
        Returns the raw parsed dictionary (mimicking a standard chat completion response).
        """
        cache_key = self._hash_input(system, messages, tools)
        
        # 1. Check Demo Cache
        if self.demo_mode:
            cached = self._read_cache(cache_key)
            if cached:
                logger.debug(f"[LLMClient] Cache HIT for {cache_key}")
                return cached
            else:
                logger.warning(f"[LLMClient] Cache MISS in DEMO_MODE for {cache_key}. Will attempt live call if keys exist.")

        # Ensure system prompt is the first message (LiteLLM handles this via the messages array)
        full_messages = [{"role": "system", "content": system}] + messages
        
        call_kwargs = {
            "model": self.full_model_str,
            "messages": full_messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "timeout": self.timeout,
        }
        if tools:
            call_kwargs["tools"] = tools
            
        # 2. Execute with Retries
        start_time = time.time()
        
        for attempt in Retrying(
            stop=stop_after_attempt(self.max_retries),
            wait=wait_exponential(multiplier=1, min=2, max=10),
            reraise=True
        ):
            with attempt:
                try:
                    logger.debug(f"[LLMClient] Attempt {attempt.retry_state.attempt_number} for {self.full_model_str}")
                    # Cast litellm response to dict for easy serialization
                    raw_response = litellm.completion(**call_kwargs)
                    response_dict = json.loads(raw_response.model_dump_json())
                    
                except Exception as e:
                    logger.error(f"[LLMClient] Call failed: {str(e)}")
                    raise e
                    
        latency = (time.time() - start_time) * 1000
        
        # 3. Log and Cache
        self._log_call(call_kwargs, response_dict, latency)
        
        if self.demo_mode:
            self._write_cache(cache_key, response_dict)
            
        return response_dict

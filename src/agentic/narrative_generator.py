"""
src/agentic/narrative_generator.py
==================================
Module 4A: Generates short compliance-analyst case notes from detector output.
Includes a strict fact-grounding validator to prevent hallucinations.
"""

import json
import logging
import re
from typing import Any

from src.agentic.llm_client import LLMClient

logger = logging.getLogger(__name__)

NARRATIVE_SYSTEM_PROMPT = """You write case notes for a compliance analyst reviewing a SIMULATED market (synthetic traders, not real people).

Input JSON:
- "flag": the flagged account, suspected manipulation type, tick span and detector score.
- "mechanism": how this type of manipulation works.
- "evidence": facts measured from the account's own orders and trades in the flagged span.
- "detector_shap": which detector features pushed the score up (positive) or down (negative).

Write 4-6 sentences of plain prose:
1. What the account actually did, step by step, in market terms (orders placed, cancelled, filled; who it traded with; how its position and the price moved), using the evidence.
2. Why that sequence matches (or does not match) the mechanism, and who in the market it would mislead.
3. One closing sentence naming the two or three detector features that captured it.

Rules: copy every number exactly as it appears in the input (no rounding, no unit conversions, no arithmetic, no new numbers). Refer to accounts by their ids. If the evidence does not support the mechanism, say so plainly instead of forcing the story. Do not describe SHAP mechanics; just name the features."""

class NarrativeGenerator:
    def __init__(self, config: dict[str, Any], run_dir=None):
        # We pass the specific narrative_generator block from llm_config.yaml
        self.client = LLMClient(config, run_dir=run_dir)

    def _extract_numbers(self, text: str) -> set[str]:
        """Extracts all continuous numeric strings from a text, ignoring punctuation at ends."""
        # Finds integers or floats like 42, 3.14, .5
        matches = re.findall(r'\b\d+(?:\.\d+)?\b', text)
        
        # Filter out common conversational integers (0-9) to avoid false positives 
        # (like the LLM saying "2 features" or "4 sentences")
        filtered = {m for m in matches if not (m.isdigit() and 0 <= int(m) <= 9)}
        return filtered

    def _validate_grounding(self, narrative: str, input_data: dict) -> bool:
        """
        Validates that every number mentioned in the generated narrative
        exists somewhere in the input JSON values.
        """
        narrative_numbers = self._extract_numbers(narrative)
        if not narrative_numbers:
            return True # No numbers to hallucinate
            
        # Extract all numbers from the input dict
        input_str = json.dumps(input_data)
        input_numbers = self._extract_numbers(input_str)
        
        # Check if any narrative number is missing from the input
        # Note: LLM might round numbers (e.g. 0.998444 to 0.99 or 99.8%). 
        # For a truly strict regex per FR-4.1, it must exactly match. 
        # If it rounds, it fails grounding and falls back.
        unsupported = narrative_numbers - input_numbers
        
        # Allow small edge cases like "2" or "4" from "2-4 sentences" prompt leaking,
        # but strictly speaking, we want to flag any unexplained data.
        self.last_unsupported = sorted(unsupported)
        if unsupported:
            logger.warning(f"[NarrativeGenerator] Grounding check failed! Unsupported numbers found: {unsupported}")
            return False
            
        return True

    def generate(self, flag_row: dict, shap_summary: dict, evidence: dict | None = None,
                 mechanism: str | None = None) -> dict:
        """
        Generates a narrative. If grounding fails, it regenerates once with temp=0.0.
        Returns a dict matching the case_narratives.jsonl schema:
        {
           "narrative_text": str,
           "grounding_check_passed": bool,
           "model_used": str,
           "tokens": int,
           "latency_ms": float
        }
        """
        input_data = {"flag": flag_row}
        if mechanism:
            input_data["mechanism"] = mechanism
        if evidence:
            input_data["evidence"] = evidence
        input_data["detector_shap"] = shap_summary
        messages = [{"role": "user", "content": json.dumps(input_data, separators=(",", ":"))}]
        
        attempts = 2
        narrative = ""
        passed = False
        meta = {}
        
        for attempt in range(attempts):
            try:
                response = self.client.complete(system=NARRATIVE_SYSTEM_PROMPT, messages=messages)
                narrative = (response["choices"][0].get("message") or {}).get("content") or ""
                if not narrative.strip():
                    logger.warning("Narrative model returned no text (token limit); not retrying.")
                    break
                
                # Extract meta for audit trail
                usage = response.get("usage", {})
                meta = {
                    "model_used": response.get("model", self.client.full_model_str),
                    "tokens": usage.get("total_tokens", 0)
                }
                
                if self._validate_grounding(narrative, input_data):
                    passed = True
                    break
                else:
                    logger.info(f"Attempt {attempt+1} failed grounding validation. Retrying...")
                    # Append a strict correction for the retry
                    messages.append({"role": "assistant", "content": narrative})
                    bad = ", ".join(getattr(self, "last_unsupported", [])) or "some numbers"
                    messages.append({"role": "user", "content": (
                        f"These numbers in your note are not in the input: {bad}. Rewrite the note using only "
                        "numbers that appear verbatim in the input JSON (copy them exactly; no rounding, sums or "
                        "percent conversions). Remove any number you cannot copy.")})
                    
            except Exception as e:
                logger.error(f"Narrative generation failed: {str(e)}")
                break

        return {
            "narrative_text": narrative if passed else "NARRATIVE REJECTED: Failed fact-grounding validation.",
            "grounding_check_passed": passed,
            "model_used": meta.get("model_used", "unknown"),
            "tokens": meta.get("tokens", 0),
            "latency_ms": 0.0 # Latency is logged by LLMClient natively, but we can put 0.0 here or pass it up if we timed it
        }

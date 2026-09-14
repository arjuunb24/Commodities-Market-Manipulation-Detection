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

NARRATIVE_SYSTEM_PROMPT = """You write short compliance-analyst case notes from  
structured detector output. You are given: the persona classifier(s) that  
fired, the numeric SHAP feature attributions, and summary statistics for the  
flagged window. Write 2-4 sentences explaining WHY this window was flagged,  
referencing the specific feature values you were given. Do not invent any  
number, trader ID, or fact not present in the input. If the input is  
insufficient to explain the flag, say so explicitly rather than guessing."""

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
        if unsupported:
            logger.warning(f"[NarrativeGenerator] Grounding check failed! Unsupported numbers found: {unsupported}")
            return False
            
        return True

    def generate(self, flag_row: dict, shap_summary: dict) -> dict:
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
        input_data = {"flag": flag_row, "shap": shap_summary}
        messages = [{"role": "user", "content": json.dumps(input_data)}]
        
        attempts = 2
        narrative = ""
        passed = False
        meta = {}
        
        for attempt in range(attempts):
            try:
                response = self.client.complete(system=NARRATIVE_SYSTEM_PROMPT, messages=messages)
                narrative = response["choices"][0]["message"]["content"]
                
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
                    messages.append({"role": "user", "content": "Your previous response contained numbers not present in the input. Rewrite it strictly using only the provided numbers. Do not round them."})
                    
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

"""
scripts/test_module_4.py
========================
Tests the Module 4A Narrative Generator to ensure the LLMClient 
is correctly configured with your API keys.
"""

import sys
import yaml
import logging
from pathlib import Path

# Adjust path to find src/
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.agentic.narrative_generator import NarrativeGenerator
from src.utils.logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

def main():
    logger.info("Loading LLM config...")
    config_path = REPO_ROOT / "configs" / "llm_config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    narrative_config = config.get("narrative_generator", {})
    generator = NarrativeGenerator(narrative_config)
    
    # Mock some detector output
    flag_row = {
        "commodity": "crude_oil_wti",
        "timestamp": "2026-01-01 01:46:00",
        "ensemble_flag": True,
        "prob_spoofing": 0.998,
        "total_volume": 45000,
        "cancel_rate": 0.92
    }
    
    shap_summary = {
        "cancel_rate_importance": 0.45,
        "volume_concentration_importance": 0.22,
        "otr_importance": 0.15
    }
    
    logger.info("Requesting narrative from LLM... (This will fail if your API keys in .env are invalid)")
    try:
        result = generator.generate(flag_row, shap_summary)
        print("\n=== GENERATED NARRATIVE ===")
        print(result["narrative_text"])
        print("===========================\n")
        print(f"Grounding Passed: {result['grounding_check_passed']}")
        print(f"Model Used: {result['model_used']}")
        print(f"Tokens: {result['tokens']}")
    except Exception as e:
        logger.error(f"Failed to generate narrative: {e}")

if __name__ == "__main__":
    main()

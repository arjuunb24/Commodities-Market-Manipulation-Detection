"""
scripts/run_adversarial_loop.py
===============================
Runs a demonstration of the Adversarial Strategist (Module 4B).
The LLM will use tools to analyze baseline metrics and propose 
mutations to the wash_trading or spoofing persona.
"""

import sys
import yaml
import logging
from pathlib import Path

# Adjust path to find src/
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.agentic.adversarial_strategist import AdversarialStrategist
from src.utils.logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

def main():
    logger.info("Loading LLM config...")
    config_path = REPO_ROOT / "configs" / "llm_config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    strategist_config = config.get("adversarial_strategist", {})
    
    # Initialize the Agent
    strategist = AdversarialStrategist(strategist_config)
    
    # Let's target the spoofing persona for round 1
    target_persona = "spoofing"
    round_num = 1
    
    logger.info("==================================================")
    logger.info(f"🚀 Launching Adversarial Strategist for {target_persona.upper()}")
    logger.info("==================================================")
    
    # Run the loop! The LLM will call tools (get_round_metrics, get_shap_drift, propose_mutation)
    result = strategist.run_strategist_round(round_num=round_num, persona=target_persona)
    
    logger.info("\n=== STRATEGIST RESULTS ===")
    import json
    print(json.dumps(result, indent=2))
    logger.info("==========================")
    logger.info("Check configs/persona_config.yaml to see if the LLM successfully mutated the file!")

if __name__ == "__main__":
    main()

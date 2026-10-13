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

import argparse

def main():
    parser = argparse.ArgumentParser(description="Run Adversarial Strategist loop.")
    parser.add_argument("--persona", type=str, default="spoofing", help="Persona to mutate ('spoofing', 'wash_trading', 'pump_and_dump', or 'all')")
    parser.add_argument("--round", type=int, default=1, help="Round number to mutate for (e.g., 1)")
    args = parser.parse_args()

    logger.info("Loading LLM config...")
    config_path = REPO_ROOT / "configs" / "llm_config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    strategist_config = config.get("adversarial_strategist", {})
    
    # Initialize the Agent
    strategist = AdversarialStrategist(strategist_config)
    
    if args.persona == "all":
        target_personas = ["spoofing", "wash_trading", "pump_and_dump"]
    else:
        target_personas = [args.persona]
    round_num = args.round
    
    logger.info("==================================================")
    logger.info(f"[*] Launching Adversarial Strategist for {', '.join(p.upper() for p in target_personas)}")
    logger.info("==================================================")
    
    # Run the loop! The LLM will call tools (get_round_metrics, get_shap_drift, propose_mutation)
    result = strategist.run_strategist_round(round_num=round_num, personas=target_personas)
    
    logger.info("\n=== STRATEGIST RESULTS ===")
    import json
    print(json.dumps(result, indent=2))
    logger.info("==========================")
    logger.info("Check configs/persona_config.yaml to see if the LLM successfully mutated the file!")

if __name__ == "__main__":
    main()

import os
import sys
import yaml
from pathlib import Path
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

load_dotenv()

from src.agentic.llm_client import LLMClient, get_gemini_pool
from src.agentic.tools import ToolRegistry

def test_rotation_and_tools():
    print("=== 1. Testing Gemini Key Pool Rotation ===")
    pool = get_gemini_pool()
    print(f"Total keys in pool: {pool.total_keys}")
    
    config_path = REPO_ROOT / "configs" / "llm_config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    strategist_cfg = config.get("adversarial_strategist", {})
    
    client = LLMClient(strategist_cfg)
    
    # Run 4 calls and verify key rotation
    for i in range(4):
        key = pool.get_next_key()
        masked = key[:6] + "..." + key[-4:] if key else "None"
        print(f"Call {i+1} picked key: {masked}")
        
    print("\n=== 2. Testing get_enforcement_notes Extraction ===")
    tools = ToolRegistry()
    for persona in ["spoofing", "wash_trading", "pump_and_dump"]:
        notes = tools.get_enforcement_notes(persona)
        first_line = notes.splitlines()[0] if notes else "EMPTY"
        print(f"[{persona}] -> {first_line} (total lines: {len(notes.splitlines())})")

if __name__ == "__main__":
    test_rotation_and_tools()

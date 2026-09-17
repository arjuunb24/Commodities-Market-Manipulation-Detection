import os
import sys
import yaml
from pathlib import Path

# Ensure src is in PYTHONPATH
sys.path.append(str(Path(__file__).parent.parent))

from src.agentic.llm_client import LLMClient
from dotenv import load_dotenv

def test_strategist_llm():
    load_dotenv()
    
    config_path = Path(__file__).parent.parent / "configs" / "llm_config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
        
    strategist_cfg = config.get("adversarial_strategist", {})
    client = LLMClient(strategist_cfg)
    
    print(f"Testing client with provider: {client.provider}, model: {client.full_model_str}")
    
    try:
        response = client.complete(
            system="You are an AI assistant. Say hello.",
            messages=[{"role": "user", "content": "Hello!"}]
        )
        print("\nSUCCESS! Response:")
        print(response.get("choices", [{}])[0].get("message", {}).get("content", ""))
    except Exception as e:
        print(f"\nFAILED: {e}")

if __name__ == "__main__":
    test_strategist_llm()

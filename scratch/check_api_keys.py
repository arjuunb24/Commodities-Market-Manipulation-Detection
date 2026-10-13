import os
import sys
from pathlib import Path
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
load_dotenv()

import litellm

def test_keys():
    keys = []
    
    # Collect all unique keys
    primary = os.getenv("GEMINI_API_KEY")
    if primary:
        keys.append(("GEMINI_API_KEY", primary.strip()))
        
    strat = os.getenv("GEMINI_STRATEGIST_API_KEY")
    if strat and (not keys or strat.strip() != keys[0][1]):
        keys.append(("GEMINI_STRATEGIST_API_KEY", strat.strip()))
        
    general = os.getenv("GEMINI_GENERAL_API_KEYS")
    if general:
        for idx, k in enumerate(general.split(",")):
            k = k.strip()
            if k and not any(k == existing[1] for existing in keys):
                keys.append((f"GEMINI_GENERAL_API_KEYS[{idx}]", k))
                
    print(f"Total unique keys detected: {len(keys)}")
    
    models_to_test = ["gemini/gemini-2.0-flash", "gemini/gemini-2.5-flash", "gemini/gemini-1.5-flash"]
    
    for name, key in keys:
        masked = key[:6] + "..." + key[-4:]
        print(f"\n--- Testing Key: {name} ({masked}) ---")
        
        # Test on gemini-2.0-flash first
        for model in ["gemini/gemini-2.0-flash"]:
            try:
                resp = litellm.completion(
                    model=model,
                    messages=[{"role": "user", "content": "Ping. Reply with one word: Pong."}],
                    api_key=key,
                    timeout=15,
                    max_retries=0
                )
                reply = resp.choices[0].message.content.strip()
                print(f"  [{model}] SUCCESS: {reply}")
            except Exception as e:
                print(f"  [{model}] FAILED: {type(e).__name__}: {e}")

if __name__ == "__main__":
    test_keys()

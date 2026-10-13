import os
import json
import urllib.request
import urllib.error
from dotenv import load_dotenv

load_dotenv()

def test_keys_raw():
    keys = []
    primary = os.getenv("GEMINI_API_KEY")
    if primary:
        keys.append(("GEMINI_API_KEY", primary.strip()))
        
    strat = os.getenv("GEMINI_STRATEGIST_API_KEY")
    if strat and strat.strip() != primary.strip():
        keys.append(("GEMINI_STRATEGIST_API_KEY", strat.strip()))
        
    general = os.getenv("GEMINI_GENERAL_API_KEYS")
    if general:
        for idx, k in enumerate(general.split(",")):
            k = k.strip()
            if k and not any(k == existing[1] for existing in keys):
                keys.append((f"GEMINI_GENERAL_API_KEYS[{idx}]", k))
                
    print(f"Total unique keys detected: {len(keys)}\n")
    
    # We test with gemini-2.0-flash via Google Generative Language REST API
    # Model endpoint: https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key=...
    model_name = "gemini-2.0-flash"
    
    for name, key in keys:
        masked = key[:6] + "..." + key[-4:]
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={key}"
        payload = {
            "contents": [{
                "parts": [{"text": "Ping. Reply with one word: Pong."}]
            }]
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"}
        )
        
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                res_body = json.loads(resp.read().decode())
                text = res_body.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "").strip()
                print(f"[SUCCESS] {name} ({masked}): {text}")
        except urllib.error.HTTPError as e:
            err_body = e.read().decode()
            print(f"[HTTP {e.code}] {name} ({masked}): {err_body[:300]}")
        except Exception as e:
            print(f"[ERROR] {name} ({masked}): {e}")

if __name__ == "__main__":
    test_keys_raw()

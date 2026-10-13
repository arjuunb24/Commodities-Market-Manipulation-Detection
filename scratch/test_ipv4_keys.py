import os
import sys
import socket
import json
import urllib.request
import urllib.error
from dotenv import load_dotenv

# Force IPv4
old_getaddrinfo = socket.getaddrinfo
def new_getaddrinfo(*args, **kwargs):
    responses = old_getaddrinfo(*args, **kwargs)
    return [r for r in responses if r[0] == socket.AF_INET]
socket.getaddrinfo = new_getaddrinfo

load_dotenv()

def test_keys():
    primary = os.getenv("GEMINI_API_KEY", "").strip()
    general = [k.strip() for k in os.getenv("GEMINI_GENERAL_API_KEYS", "").split(",") if k.strip()]
    
    all_keys = [primary] + [k for k in general if k != primary]
    print(f"Testing {len(all_keys)} unique keys with forced IPv4...")
    
    models = ["gemini-3.6-flash", "gemini-3.8-flash"]
    
    for idx, key in enumerate(all_keys):
        masked = key[:6] + "..." + key[-4:]
        print(f"\n--- Key {idx+1}: {masked} ---")
        for model in models:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"
            payload = json.dumps({"contents": [{"parts": [{"text": "Reply with only: PONG"}]}]}).encode("utf-8")
            req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
            
            try:
                with urllib.request.urlopen(req, timeout=5) as resp:
                    data = json.loads(resp.read().decode())
                    text = data["candidates"][0]["content"]["parts"][0]["text"].strip()
                    print(f"  [{model}] SUCCESS: {text}")
            except urllib.error.HTTPError as e:
                err = e.read().decode()
                print(f"  [{model}] HTTP {e.code}: {err[:200]}")
            except Exception as e:
                print(f"  [{model}] ERROR: {e}")

if __name__ == "__main__":
    test_keys()

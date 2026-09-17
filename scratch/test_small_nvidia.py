import os
import json
import urllib.request
from dotenv import load_dotenv

load_dotenv()
api_key = os.environ.get("NVIDIA_NIM_API_KEY")

models_to_test = [
    "meta/llama-3.1-8b-instruct",
    "meta/llama3-8b-instruct",
    "mistralai/mistral-7b-instruct-v0.3",
    "google/gemma-2-9b-it"
]

print("Testing smaller models...")
for model in models_to_test:
    url = "https://integrate.api.nvidia.com/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "Hi"}],
        "max_tokens": 5
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, 
        data=data, 
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    )
    
    try:
        with urllib.request.urlopen(req) as response:
            print(f"[WORKING] {model}")
    except urllib.error.HTTPError as e:
        print(f"[FAILED] {model} - HTTP {e.code}")
    except Exception as e:
        print(f"[FAILED] {model} - {e}")

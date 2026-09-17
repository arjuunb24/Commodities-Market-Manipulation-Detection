import os
import json
import urllib.request
from dotenv import load_dotenv

load_dotenv()

def test_nvidia_raw():
    api_key = os.environ.get("NVIDIA_NIM_API_KEY")
    url = "https://integrate.api.nvidia.com/v1/chat/completions"
    
    payload = {
        "model": "nvidia/llama-3.1-nemotron-70b-instruct",
        "messages": [{"role": "user", "content": "Hello!"}],
        "max_tokens": 50
    }
    
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, 
        data=data, 
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
    )
    
    try:
        with urllib.request.urlopen(req) as response:
            result = json.loads(response.read().decode())
            print(f"SUCCESS: {result['choices'][0]['message']['content']}")
    except urllib.error.HTTPError as e:
        body = e.read().decode()
        print(f"FAILED: {e.code} - {body}")
    except Exception as e:
        print(f"FAILED: {e}")

if __name__ == "__main__":
    test_nvidia_raw()

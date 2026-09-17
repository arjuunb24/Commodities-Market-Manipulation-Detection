import os
import json
import urllib.request
from dotenv import load_dotenv

load_dotenv()

def test_nvidia_models():
    api_key = os.environ.get("NVIDIA_NIM_API_KEY")
    url = "https://integrate.api.nvidia.com/v1/models"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {api_key}"})
    
    try:
        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read().decode())
            print("Available models:")
            for m in data.get("data", []):
                print(f"- {m['id']}")
    except Exception as e:
        print(f"Error fetching models: {e}")

if __name__ == "__main__":
    test_nvidia_models()

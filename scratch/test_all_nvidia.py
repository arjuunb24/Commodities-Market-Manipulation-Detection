import os
import json
import urllib.request
from dotenv import load_dotenv
import concurrent.futures

load_dotenv()
api_key = os.environ.get("NVIDIA_NIM_API_KEY")

def test_model(model_name):
    url = "https://integrate.api.nvidia.com/v1/chat/completions"
    payload = {
        "model": model_name,
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
            return model_name, True, "OK"
    except urllib.error.HTTPError as e:
        return model_name, False, str(e.code)
    except Exception as e:
        return model_name, False, str(e)

def main():
    url = "https://integrate.api.nvidia.com/v1/models"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {api_key}"})
    
    try:
        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read().decode())
            models = [m['id'] for m in data.get("data", [])]
    except Exception as e:
        print(f"Error fetching models: {e}")
        return

    print(f"Testing {len(models)} models...")
    working_models = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(test_model, m): m for m in models}
        for future in concurrent.futures.as_completed(futures):
            model_name, success, msg = future.result()
            if success:
                print(f"[WORKING] {model_name}")
                working_models.append(model_name)

    print("\nWorking models you can use:")
    for m in working_models:
        print(f"- {m}")

if __name__ == "__main__":
    main()

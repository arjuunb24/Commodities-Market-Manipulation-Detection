import os
from dotenv import load_dotenv
import litellm

# Load API keys from .env
load_dotenv()

def test_nvidia():
    print("Testing NVIDIA NIM via litellm...")
    try:
        response = litellm.completion(
            model="nvidia_nim/meta/llama-3.2-90b-vision-instruct",
            messages=[{"role": "user", "content": "Say hello to the NVIDIA API!"}],
            temperature=0.7,
            max_tokens=50
        )
        
        print("\nSUCCESS! Received response:")
        print("---------------------------")
        print(response.choices[0].message.content)
        print("---------------------------")
        
    except Exception as e:
        print(f"\nFAILED: {e}")

if __name__ == "__main__":
    test_nvidia()

import os
import litellm
from dotenv import load_dotenv

# Load CEREBRAS_API_KEY from .env
load_dotenv()

# The model to test (we use llama3.1-8b because it's definitely available on their free/dev tiers)
model_name = "cerebras/llama3.1-8b"

print(f"Testing Cerebras API Key (Key starts with: {os.environ.get('CEREBRAS_API_KEY', '')[:7]}...)")
print(f"Model: {model_name}\n")

try:
    response = litellm.completion(
        model=model_name,
        messages=[{"role": "user", "content": "Say 'hi' and nothing else."}],
        temperature=0.0
    )
    print("✅ SUCCESS! The API responded with:")
    print("-----------------------------------")
    print(response.choices[0].message.content)
    print("-----------------------------------")
except litellm.AuthenticationError:
    print("❌ ERROR: Wrong API Key! Please check your CEREBRAS_API_KEY in the .env file.")
except Exception as e:
    print(f"❌ ERROR: The API request failed. Reason:\n{e}")

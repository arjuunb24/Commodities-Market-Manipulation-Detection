import os
import time
import litellm
from dotenv import load_dotenv

load_dotenv()

def test_large_prompt():
    print("Preparing a massive prompt for OpenRouter...")
    
    # Create a giant system prompt (~4000+ words)
    chunk = "This is a block of text designed to test the context window. It contains standard English words and sentences to ensure the tokenizer processes it normally. We will repeat this many times. "
    large_system_prompt = chunk * 500  
    
    print(f"System prompt length: {len(large_system_prompt)} characters.")
    
    print("\nSending request to OpenRouter (meta-llama/llama-3.1-70b-instruct)...")
    start_time = time.time()
    
    try:
        response = litellm.completion(
            model="openrouter/meta-llama/llama-3.1-70b-instruct",
            messages=[
                {"role": "system", "content": large_system_prompt},
                {"role": "user", "content": "Based on the massive block of text above, what is the main purpose of the text?"}
            ],
            temperature=0.7,
            max_tokens=150
        )
        
        latency = time.time() - start_time
        print(f"\nSUCCESS! Request completed in {latency:.2f} seconds.")
        print("\nResponse:")
        print(response.get("choices", [{}])[0].get("message", {}).get("content", ""))
        
    except Exception as e:
        print(f"\nFAILED: {e}")

if __name__ == "__main__":
    test_large_prompt()

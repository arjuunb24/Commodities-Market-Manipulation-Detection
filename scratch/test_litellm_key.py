import os
import socket
import litellm
from dotenv import load_dotenv

# Force IPv4
old_getaddrinfo = socket.getaddrinfo
def new_getaddrinfo(*args, **kwargs):
    responses = old_getaddrinfo(*args, **kwargs)
    return [r for r in responses if r[0] == socket.AF_INET]
socket.getaddrinfo = new_getaddrinfo

load_dotenv()

general = [k.strip() for k in os.getenv("GEMINI_GENERAL_API_KEYS", "").split(",") if k.strip()]
print(f"Testing direct api_key parameter in litellm with key: {general[0][:6]}...{general[0][-4:]}")

resp = litellm.completion(
    model="gemini/gemini-3.6-flash",
    messages=[{"role": "user", "content": "Reply: Hello LiteLLM"}],
    api_key=general[0]
)
print("SUCCESS:", resp.choices[0].message.content)

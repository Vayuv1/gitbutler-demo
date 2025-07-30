# We need to start da_server.py first and run this client.

import requests

response = requests.post(
    "http://localhost:8000/generate",
    json={
        "model": "gemma3:1b",
        "prompt": "Explain the difference between TCP and UDP in 100 words.",
        "stream": False
    }
)

print(response.json()["response"])
# print(response.json())
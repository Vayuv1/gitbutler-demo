# We need to start dc_completion_server.py first and run this client.

import requests

response = requests.post(
    "http://localhost:8000/completion",
    json={
        "model": "gemma3",
        "prompt": "What is quantum computing?",
    }
)

print(response.json()["response"])

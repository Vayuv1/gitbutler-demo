# We need to start da_server.py first and run this client.

import requests

response = requests.post(
    "http://localhost:8000/generate-stream",
    json={
        "model": "gemma3",
        "prompt": "Explain the difference between TCP and UDP."
    },
    stream=True
)

# Iterate through the streamed response
if response.status_code == 200:
    for chunk in response.iter_content(chunk_size=None):
        if chunk:
            # Decode the chunk and print it (or process it as needed)
            print(chunk.decode('utf-8'), end='', flush=True)
else:
    print(f"Error: {response.status_code}")
    print(response.text)

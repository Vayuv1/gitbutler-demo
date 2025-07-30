import requests
import json

def stream_prompt_response(prompt_text, model="gemma3"):
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": model,
        "prompt": prompt_text,
        "stream": True
    }
    with requests.post(url, json=payload, stream=True) as response:
        if response.status_code != 200:
            raise Exception(
                f"Streaming failed: {response.status_code} {response.text}")
        for line in response.iter_lines():
            if line:
                data = json.loads(line.decode('utf-8'))
                token = data.get("response", "")
                print(token, end="", flush=True)

# Example usage
stream_prompt_response(
    "Write a short, friendly welcome message for a new user.")
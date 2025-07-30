import requests

def send_custom_prompt(prompt_text,
                       model="gemma3",
                       temperature=0.7,
                       max_tokens=150):
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": model,
        "prompt": prompt_text,
        "stream": False,
        "options": {
            "temperature": temperature,
            "max_tokens": max_tokens
        }
    }
    response = requests.post(url, json=payload)
    if response.status_code == 200:
        return response.json().get('response')
    else:
        raise Exception(f"Failed to generate response: {response.text}")

response_text = send_custom_prompt(
    "Generate a short poem about the ocean.", temperature=0.9)
print(response_text)

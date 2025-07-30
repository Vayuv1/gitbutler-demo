import requests


def send_prompt_to_ollama(prompt_text, model="gemma3"):
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": model,
        "prompt": prompt_text,
        "stream": False
    }
    try:
        # response = requests.post(url, json=payload, timeout=10)
        response = requests.post(url, json=payload, timeout=1000)
        response.raise_for_status()
        return response.json().get('response')
    except requests.RequestException as e:
        print("An error occurred:", e)
        return None

# Example usage
result = send_prompt_to_ollama("Say hello to user.")
if result:
    print(result)

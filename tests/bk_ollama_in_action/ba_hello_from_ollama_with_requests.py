import requests

# Define the Ollama API endpoint
url = "http://localhost:11434/api/generate"

# Create the payload with model name and prompt
payload = {
    "model": "gemma3",
    "prompt": "Please say hello to user.",
    "stream": False
}

# Send the POST request
response = requests.post(url, json=payload)

# Parse and print the response
if response.status_code == 200:
    generated_text = response.json().get('response')
    print("Model Response:", generated_text)
else:
    print("Error:", response.status_code)
    print("Text:", response.text)
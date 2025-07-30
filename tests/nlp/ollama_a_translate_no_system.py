import requests

# Ollama API endpoint
url = "http://localhost:11434/api/generate"

text = "cessna one two three descend and maintain five thousand turn left heading two seven zero"
    # "cessna one two three turn left heading two seven zero"
prompt = (
    f"Given an ATC transcript: '{text}' "
    "Translate to a single sentence with casing, punctuation, and correct numerical numbers. " 
    "Below are examples of translated text: "
    "- 'riddle one three five climb and maintain eleven thousand' to 'Riddle 135, climb and maintain 11,000.' "
    "- 'Eclipse one three five alpha turn right heading two three four.' to 'Eclipse 135A, turn right heading 234.'"
)

# print(prompt)

# Request data
data = {
    "model": "gemma3:4b",
    "prompt": prompt, 
    "format": "json",
    "stream": False
}


# Send POST request
response = requests.post(url, json=data)

# Print the response
if response.status_code == 200:
    print(response.json()["response"])
else:
    print(f"Error: {response.status_code}", response.text)

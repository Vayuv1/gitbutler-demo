import requests
import json

conversation_history = []

def build_conversation_prompt(history, user_input):
    prompt = ""
    for pair in history:
        prompt += f"User: {pair['user']}\nAssistant: {pair['assistant']}\n"
    prompt += f"User: {user_input}\nAssistant:"
    return prompt

def chat_with_memory(user_input, model="gemma3"):
    global conversation_history
    prompt = build_conversation_prompt(conversation_history, user_input)
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False
    }
    response = requests.post(url, json=payload)
    if response.status_code == 200:
        assistant_reply = response.json().get('response', '').strip()
        conversation_history.append({
            "user": user_input,
            "assistant": assistant_reply
        })
        return assistant_reply
    else:
        raise Exception("Request failed:", response.status_code)

def stream_with_memory(user_input, model="gemma3"):
    global conversation_history
    prompt = build_conversation_prompt(conversation_history, user_input)
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": True
    }
    with requests.post(url, json=payload, stream=True) as response:
        assistant_reply = ""
        for line in response.iter_lines():
            if line:
                data = json.loads(line.decode('utf-8'))
                token = data.get("response", "")
                assistant_reply += token
                print(token, end="", flush=True)

        conversation_history.append({
            "user": user_input,
            "assistant": assistant_reply.strip()
        })

# Example usage
print(stream_with_memory("What's the capital of Italy?"))
print(stream_with_memory("List five attractions of that city."))

import requests
import json


class OllamaClient:
    def __init__(self, model='gemma3', host='http://localhost:11434'):
        self.model = model
        self.api_url = f"{host}/api/generate"
        self.history = []   # Stores conversation turns as dictionaries

    def set_model(self, model_name):
        self.model = model_name


    def send_prompt(self, prompt_text, options=None):
        payload = {
            "model": self.model,
            "prompt": prompt_text,
            "stream": False
        }
        if options:
            payload["options"] = options

        response = requests.post(self.api_url, json=payload)
        if response.status_code == 200:
            data = response.json()
            return data.get("response", "").strip()
        else:
            raise Exception(
                f"Request failed: {response.status_code} - {response.text}")


    def stream_prompt(self, prompt_text, options=None):
        payload = {
            "model": self.model,
            "prompt": prompt_text,
            "stream": True
        }
        if options:
            payload["options"] = options

        with requests.post(self.api_url, json=payload,
                           stream=True) as response:
            if response.status_code != 200:
                raise Exception(f"Streaming failed: \
                                {response.status_code} - {response.text}")

            for line in response.iter_lines():
                if line:
                    data = json.loads(line.decode('utf-8'))
                    token = data.get("response", "")
                    print(token, end="", flush=True)


    def build_prompt_with_history(self, user_input):
        prompt = ""
        for exchange in self.history:
            prompt += f"User: {exchange['user']}\n"
            prompt += f"Assistant: {exchange['assistant']}\n"
        prompt += f"User: {user_input}\nAssistant:"
        return prompt


    def chat(self, user_input, stream=False, options=None):
        full_prompt = self.build_prompt_with_history(user_input)
        if stream:
            print("Assistant: ", end="", flush=True)
            assistant_reply = ""
            with requests.post(
                self.api_url,
                json={
                    "model": self.model,
                    "prompt": full_prompt,
                    "stream": True,
                    "options": options or {}
                },
                stream=True
            ) as response:
                for line in response.iter_lines():
                    if line:
                        data = json.loads(line.decode('utf-8'))
                        token = data.get("response", "")
                        assistant_reply += token
                        print(token, end="", flush=True)
                print()
        else:
            assistant_reply = self.send_prompt(full_prompt, options)

        self.history.append({
            "user": user_input,
            "assistant": assistant_reply.strip()
        })

        return assistant_reply.strip()


ollama_bot = OllamaClient()

# Single prompt
response = ollama_bot.send_prompt("What is reinforcement learning?")
print("Response:", response)

# Chat with memory
ollama_bot.chat("Who was Alan Turing?")
ollama_bot.chat("What was his contribution to computer science?", stream=True)

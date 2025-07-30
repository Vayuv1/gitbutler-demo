# To run this code: uvicorn dd_chat_server:app --reload
#
# To test this code, run the following in CLI one by one:
# curl -X POST "http://localhost:8000/chat" -H "Content-Type: application/json" -d '{"session_id": "user123", "user_input": "Where is the Eiffel Tower?"}'
# curl -X POST "http://localhost:8000/chat" -H "Content-Type: application/json" -d '{"session_id": "user123", "user_input": "How tall is it?"}'

from typing import List
from fastapi import FastAPI
from pydantic import BaseModel
import requests

app = FastAPI()
conversation_memory = {}

class ChatRequest(BaseModel):
    session_id: str
    user_input: str
    model: str = "gemma3"
    stream: bool = False
    options: dict = None

def build_chat_prompt(history: List[dict], user_input: str):
    prompt = ""
    for turn in history:
        prompt += f"User: {turn['user']}\nAssistant: {turn['assistant']}\n"
    prompt += f"User: {user_input}\nAssistant:"
    return prompt

@app.post("/chat")
def chat_with_memory(request: ChatRequest):
    history = conversation_memory.get(request.session_id, [])
    full_prompt = build_chat_prompt(history, request.user_input)
    payload = {
        "model": request.model,
        "stream": request.stream,
        "prompt": full_prompt,
        "options": request.options or {}
    }

    response = requests.post("http://localhost:11434/api/generate",
                             json=payload)

    if response.status_code != 200:
        return {"error": response.text}

    assistant_reply = response.json().get("response", "").strip()

    # Update memory
    history.append({"user": request.user_input, "assistant": assistant_reply})
    conversation_memory[request.session_id] = history

    return {"response": assistant_reply}
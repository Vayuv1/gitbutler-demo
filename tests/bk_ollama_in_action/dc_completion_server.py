# To run this code: uvicorn dc_completion_server:app --reload
#
# To test this code, run the dc_completion_client.py or the following in CLI:
# curl -X POST "http://localhost:8000/completion" -H "Content-Type: application/json" -d '{"prompt": "Say Hello to user."}'

from fastapi import FastAPI
from pydantic import BaseModel
import requests

app = FastAPI()

class CompletionRequest(BaseModel):
    prompt: str
    model: str = "gemma3"
    options: dict = None

@app.post("/completion")
def generate_completion(request: CompletionRequest):
    payload = {
        "model": request.model,
        "prompt": request.prompt,
        "stream": False,
        "options": request.options or {}
    }

    response = requests.post(
        "http://localhost:11434/api/generate", json=payload)

    if response.status_code != 200:
        return {"error": response.text}

    return {"response": response.json().get("response", "")}
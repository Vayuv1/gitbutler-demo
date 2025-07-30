# To run this code: uvicorn da_server:app --reload

import json
from fastapi.responses import StreamingResponse
from fastapi import FastAPI, Request
from pydantic import BaseModel
import requests

app = FastAPI()
OLLAMA_URL = "http://localhost:11434/api/generate"

class PromptRequest(BaseModel):
    model: str
    prompt: str
    stream: bool = False
    options: dict = None

@app.post("/generate")
def generate_completion(request_data: PromptRequest):
    payload = {
        "model": request_data.model,
        "prompt": request_data.prompt,
        "stream": False
    }
    if request_data.options:
        payload["options"] = request_data.options
    if request_data.stream:
        payload["stream"] = True    # ignored in this sync implementation

    response = requests.post(OLLAMA_URL, json=payload)

    if response.status_code != 200:
        return {"error": "Ollama request failed",
                "status": response.status_code}

    return {"response": response.json().get("response", "")}


@app.post("/generate-stream")
def generate_stream(request_data: PromptRequest):
    def stream_response():
        payload = {
            "model": request_data.model,
            "prompt": request_data.prompt,
            "stream": True
        }
        if request_data.options:
            payload["options"] = request_data.options

        with requests.post(OLLAMA_URL, json=payload, stream=True) as response:
            for line in response.iter_lines():
                if line:
                    data = json.loads(line.decode("utf-8"))
                    token = data.get("response", "")
                    yield token

    return StreamingResponse(stream_response(), media_type="text/plain")

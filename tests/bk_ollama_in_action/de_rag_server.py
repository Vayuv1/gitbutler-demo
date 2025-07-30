# To run this code: uvicorn de_rag_server:app --reload
#
# To test this code, run the following in CLI:
# curl -X POST "http://localhost:8000/rag" -H "Content-Type: application/json" -d '{"query": "What is quantum computing?"}'

from fastapi import FastAPI
from pydantic import BaseModel
import requests

app = FastAPI()

# Simulate a very basic retrieval system
knowledge_base = {
    "quantum computing": "Quantum computing uses quantum bits, or qubits, which can represent 0 and 1 simultaneously.",
    "machine learning": "Machine learning is a field of AI that enables systems to learn patterns from data."
}

class RAGRequest(BaseModel):
    query: str
    model: str = "gemma3"
    stream: bool = False

def retrieve_context(query):
    for key, value in knowledge_base.items():
        if key in query.lower():
            return value

    return ""

@app.post("/rag")
def rag_search(request: RAGRequest):
    context = retrieve_context(request.query)
    prompt = f"Context: {context}\n\nUser Question: {request.query}\n\nAnswer:"
    payload = {
        "model": request.model,
        "stream": request.stream,
        "prompt": prompt
    }

    response = requests.post("http://localhost:11434/api/generate", json=payload)

    if response.status_code != 200:
        return {"error": response.text}

    return {"response": response.json()["response"]}
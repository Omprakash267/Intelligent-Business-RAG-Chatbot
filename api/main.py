from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional, Dict, Any

import chatbot  # uses your existing retrieval + ollama
from sentiment import analyze_sentiment

app = FastAPI(title="Business Chatbot API", version="1.0")

class ChatRequest(BaseModel):
    query: str
    k: int = 4
    model: Optional[str] = None

class ChatResponse(BaseModel):
    answer: str
    sentiment: Dict[str, Any]

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    model = req.model or chatbot.OLLAMA_MODEL
    answer = chatbot.answer(req.query, k=req.k, model=model)
    sentiment = analyze_sentiment(answer)
    return {"answer": answer, "sentiment": sentiment}

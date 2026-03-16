from typing import Dict, Any
from transformers import pipeline

# Loaded once (fast after first load)
_sentiment = pipeline("sentiment-analysis")

def analyze_sentiment(text: str) -> Dict[str, Any]:
    """
    Returns: {"label": "POSITIVE"/"NEGATIVE", "score": float}
    """
    if not text or not text.strip():
        return {"label": "NEUTRAL", "score": 0.0}

    result = _sentiment(text[:512])[0]  # keep it safe/fast
    return {"label": result["label"], "score": float(result["score"])}

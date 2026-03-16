import re
import requests
from typing import Any, Dict, List, Set, Tuple

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from sentiment import analyze_sentiment

# ---- Config ----
EMBED_MODEL = "sentence-transformers/all-mpnet-base-v2"
OLLAMA_MODEL = "llama2:latest"
DB_DIR = "embeddings"
OLLAMA_URL = "http://localhost:11434/api/generate"

# Guardrail tuning
MIN_CONTEXT_CHARS = 200
MIN_QUERY_CTX_OVERLAP = 1

# Professional refusal message (guardrail response)
REFUSAL_MESSAGE = (
    "I'm sorry, but I cannot find a reliable answer in the provided documents. "
    "Please check the source material or rephrase your question."
)

# ---- Load once (performance) ----
_embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
_db = Chroma(persist_directory=DB_DIR, embedding_function=_embeddings)


# -------------------------
# Guardrail helpers
# -------------------------
def _extract_keywords(text: str) -> Set[str]:
    words = re.findall(r"[a-zA-Z0-9]+", (text or "").lower())
    stop = {
        "the", "is", "are", "a", "an", "of", "to", "in", "for", "and", "or", "on",
        "with", "what", "who", "when", "where", "how", "can", "could", "should",
        "would", "please", "tell", "me", "about", "give"
    }
    return {w for w in words if w not in stop and len(w) > 2}


def _context_is_sufficient(context: str, min_chars: int = MIN_CONTEXT_CHARS) -> bool:
    return bool(context and len(context.strip()) >= min_chars)


def _keyword_overlap_ok(query: str, context: str, min_overlap: int = MIN_QUERY_CTX_OVERLAP) -> bool:
    qk = _extract_keywords(query)
    if not qk:
        return True
    ctx = (context or "").lower()
    overlap = sum(1 for w in qk if w in ctx)
    return overlap >= min_overlap


def _post_validate_answer(generated: str, context: str) -> str:
    """
    Post-check: if answer is not grounded in context, return polite refusal.
    """
    if not generated or not generated.strip():
        return REFUSAL_MESSAGE

    ans = generated.strip()

    if ans.lower().replace(".", "") == "i don't know":
        return REFUSAL_MESSAGE

    ans_kw = _extract_keywords(ans)
    if not ans_kw:
        return ans

    ctx = (context or "").lower()
    overlap = sum(1 for w in ans_kw if w in ctx)

    if overlap < 1:
        return REFUSAL_MESSAGE

    return ans


def _make_snippet(text: str, max_len: int = 180) -> str:
    clean = re.sub(r"\s+", " ", (text or "").strip())
    if len(clean) <= max_len:
        return clean
    return clean[: max_len - 3].rstrip() + "..."


def _confidence_from_retrieval(query: str, docs_scores: List[Tuple[Any, float]]) -> Dict[str, Any]:
    if not docs_scores:
        return {"label": "low", "score": 0.0}

    top_distance = float(docs_scores[0][1])
    qk = _extract_keywords(query)
    top_text = docs_scores[0][0].page_content if docs_scores[0][0] else ""

    if qk:
        top_lc = top_text.lower()
        overlap = sum(1 for w in qk if w in top_lc)
        overlap_ratio = overlap / max(len(qk), 1)
    else:
        overlap_ratio = 1.0

    # Chroma returns a distance where lower is better.
    dist_points = 2 if top_distance <= 0.40 else (1 if top_distance <= 0.90 else 0)
    overlap_points = 2 if overlap_ratio >= 0.50 else (1 if overlap_ratio >= 0.20 else 0)
    total = dist_points + overlap_points

    if total >= 3:
        label = "high"
    elif total >= 2:
        label = "medium"
    else:
        label = "low"

    score = round(total / 4.0, 2)
    return {"label": label, "score": score}


# -------------------------
# Retrieval
# -------------------------
def retrieve_docs(query: str, k: int = 4) -> List:
    return _db.similarity_search(query, k=k)


def retrieve_docs_with_scores(query: str, k: int = 4) -> List[Tuple[Any, float]]:
    return _db.similarity_search_with_score(query, k=k)


def retrieve_context(query: str, k: int = 4) -> str:
    docs = retrieve_docs(query, k=k)
    return "\n\n".join(d.page_content for d in docs)


# -------------------------
# Ollama
# -------------------------
def ask_ollama(prompt: str, model: str = OLLAMA_MODEL, timeout: int = 120) -> str:
    r = requests.post(
        OLLAMA_URL,
        json={"model": model, "prompt": prompt, "stream": False},
        timeout=timeout,
    )
    r.raise_for_status()
    return r.json().get("response", "").strip()


# -------------------------
# RAG Answer with Guardrails
# -------------------------
def answer(query: str, k: int = 4, model: str = OLLAMA_MODEL) -> str:
    context = retrieve_context(query, k=k).strip()

    # Guardrail 1 - insufficient context
    if not _context_is_sufficient(context):
        return REFUSAL_MESSAGE

    # Guardrail 2 - irrelevant context
    if not _keyword_overlap_ok(query, context):
        return REFUSAL_MESSAGE

    # Guardrail 3 - strict prompt rules
    prompt = f"""
You are a business support assistant.

STRICT RULES (must follow):
1) Use ONLY the information in CONTEXT.
2) If the answer is not explicitly present in CONTEXT, reply with a polite refusal message.
3) Do NOT use outside knowledge or assumptions.
4) Keep the answer clear and concise.

CONTEXT:
{context}

QUESTION:
{query}

ANSWER:
"""

    raw = ask_ollama(prompt, model=model)

    # Guardrail 4 - post validation
    return _post_validate_answer(raw, context)


def answer_with_sources(query: str, k: int = 4, model: str = OLLAMA_MODEL) -> Dict[str, Any]:
    docs_scores = retrieve_docs_with_scores(query, k=k)
    docs = [doc for doc, _ in docs_scores]
    context = "\n\n".join(d.page_content for d in docs).strip()

    confidence = _confidence_from_retrieval(query, docs_scores)
    sources = []
    for i, (doc, distance) in enumerate(docs_scores[:2], start=1):
        source_name = (doc.metadata or {}).get("source", "knowledge.txt")
        sources.append(
            {
                "rank": i,
                "source": source_name,
                "distance": round(float(distance), 4),
                "snippet": _make_snippet(doc.page_content),
            }
        )

    if not _context_is_sufficient(context):
        return {"answer": REFUSAL_MESSAGE, "confidence": confidence, "sources": sources}

    if not _keyword_overlap_ok(query, context):
        return {"answer": REFUSAL_MESSAGE, "confidence": confidence, "sources": sources}

    prompt = f"""
You are a business support assistant.

STRICT RULES (must follow):
1) Use ONLY the information in CONTEXT.
2) If the answer is not explicitly present in CONTEXT, reply with a polite refusal message.
3) Do NOT use outside knowledge or assumptions.
4) Keep the answer clear and concise.

CONTEXT:
{context}

QUESTION:
{query}

ANSWER:
"""

    raw = ask_ollama(prompt, model=model)
    text = _post_validate_answer(raw, context)
    return {"answer": text, "confidence": confidence, "sources": sources}


# -------------------------
# With Sentiment
# -------------------------
def answer_with_sentiment(query: str, k: int = 4, model: str = OLLAMA_MODEL) -> Dict[str, Any]:
    enriched = answer_with_sources(query, k=k, model=model)
    text = enriched["answer"]
    sent = analyze_sentiment(text)
    return {
        "answer": text,
        "sentiment": sent,
        "confidence": enriched["confidence"],
        "sources": enriched["sources"],
    }

import os
import sys
import requests
from pathlib import Path

def fail(msg: str, code: int = 1):
    print(f"[FAIL] {msg}")
    sys.exit(code)

def ok(msg: str):
    print(f"[OK] {msg}")

def main():
    # 1) Python version
    if sys.version_info < (3, 10):
        fail("Python >= 3.10 required")

    ok(f"Python {sys.version.split()[0]}")

    # 2) Ollama health
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=3)
        if r.status_code != 200:
            fail(f"Ollama reachable but unexpected status: {r.status_code}")
        ok("Ollama is reachable on 127.0.0.1:11434")
    except Exception as e:
        fail(f"Ollama not reachable on 127.0.0.1:11434. Start it with: `ollama serve` ({e})")

    # 3) Chroma DB directory
    db_dir = Path("embeddings")
    if not db_dir.exists():
        fail("Chroma embeddings/ folder not found. Run: `python rag_build.py`")
    ok("Chroma embeddings/ folder exists")

    # 4) Quick import check
    try:
        import chatbot
        ok("Imports OK (chatbot)")
    except Exception as e:
        fail(f"Import error in chatbot.py: {e}")

    print("\nAll checks passed.")

if __name__ == "__main__":
    main()

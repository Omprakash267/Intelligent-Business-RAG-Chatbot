import os
from pathlib import Path
from typing import List

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

# ---- Config ----
DATA_DIR = Path("data")
DB_DIR = "embeddings"
EMBED_MODEL = "sentence-transformers/all-mpnet-base-v2"

CHUNK_SIZE = 900
CHUNK_OVERLAP = 150


def load_txt_files(data_dir: Path) -> List[Document]:
    """
    Load all .txt files from data_dir into LangChain Document objects.
    """
    docs: List[Document] = []
    if not data_dir.exists():
        raise FileNotFoundError(f"Missing data directory: {data_dir.resolve()}")

    txt_files = sorted(data_dir.glob("*.txt"))
    if not txt_files:
        raise FileNotFoundError(
            f"No .txt files found in {data_dir.resolve()}. "
            f"Create one like data/knowledge.txt"
        )

    for fp in txt_files:
        text = fp.read_text(encoding="utf-8", errors="ignore").strip()
        if not text:
            continue
        docs.append(Document(page_content=text, metadata={"source": str(fp.name)}))

    if not docs:
        raise ValueError("All .txt files were empty.")
    return docs


def build_chroma():
    # Load
    raw_docs = load_txt_files(DATA_DIR)
    print(f"Loaded {len(raw_docs)} document(s) from {DATA_DIR.resolve()}")

    # Split
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(raw_docs)
    print(f"Split into {len(chunks)} chunk(s)")

    # Embeddings
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    # Rebuild DB from scratch each run (clean + deterministic)
    if os.path.exists(DB_DIR):
        # If you prefer incremental updates, remove this delete logic
        # and just call Chroma(persist_directory=..., embedding_function=...).add_documents(...)
        import shutil
        shutil.rmtree(DB_DIR)

    db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_DIR,
    )

    # Persist
    db.persist()
    print(f"✅ Chroma DB saved to: {Path(DB_DIR).resolve()}")


if __name__ == "__main__":
    build_chroma()

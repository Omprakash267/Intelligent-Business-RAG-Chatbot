# Intelligent Business RAG Chatbot

An **Intelligent Retrieval-Augmented Generation (RAG) Chatbot** designed to answer queries using information retrieved from local business documents. The system integrates **vector search, local LLM inference, hallucination guardrails, sentiment analysis, and dual interfaces (UI and API)** to deliver reliable and explainable responses.

---

## Project Overview

Many traditional chatbots rely only on the language model’s internal knowledge, which can lead to **hallucinated or inaccurate answers**. This project addresses that issue by implementing a **Retrieval-Augmented Generation (RAG) architecture**.

The chatbot retrieves relevant information from locally stored business documents using a **vector database**, then provides answers grounded in those documents using a **local LLM**.

Key capabilities include:

* Context-based response generation
* Multi-layer hallucination detection
* Sentiment analysis for responses
* Configurable document retrieval
* Source citation with confidence scoring
* Streamlit UI and FastAPI API access

---

## Features

* Retrieval-Augmented Generation (RAG)
* Local vector database using Chroma
* Local LLM inference using Ollama
* Multi-layer hallucination guardrails
* Sentiment analysis for every response
* Adjustable retrieval parameters (Top-K)
* Dynamic model selection
* Source citation with grounding confidence
* Dual interface:

  * Streamlit web UI
  * FastAPI REST API

---

## Project Structure

```
business_chatbot/
│
├── app.py                # Streamlit user interface
├── chatbot.py            # Core RAG chatbot logic
├── sentiment.py          # Sentiment analysis module
├── rag_build.py          # Document embedding and vector DB creation
│
├── api/
│   └── main.py           # FastAPI endpoint
│
├── embeddings/           # Chroma vector database
├── data/                 # Business documents (text, PDFs)
│
├── scripts/
│   ├── setup.ps1
│   └── validate.ps1
│
└── README.md
```

---

## System Architecture

User Query
↓
Streamlit UI / FastAPI API
↓
Retriever (Chroma Vector Database)
↓
Context Builder
↓
Ollama Local LLM
↓
Hallucination Guardrails
↓
Sentiment Analysis
↓
Final Response with Confidence Score and Sources

---

## Installation

### 1. Clone the Repository

```
git clone <repository_url>
cd business_chatbot
```

---

### 2. Create a Virtual Environment

```
python -m venv .venv
```

Activate the environment.

Windows:

```
.\.venv\Scripts\activate
```

Linux/Mac:

```
source .venv/bin/activate
```

---

### 3. Install Dependencies

```
pip install -r requirements.txt
```

If the requirements file is not available:

```
pip install streamlit fastapi uvicorn chromadb langchain sentence-transformers transformers requests
```

---

## Ollama Setup

Install Ollama from:

https://ollama.com

Pull a model:

```
ollama pull llama2
```

Ollama runs locally at:

```
http://localhost:11434
```

---

## Building the Vector Database

Place your business documents inside the **data/** folder.

Then run:

```
python rag_build.py
```

This script processes the documents, generates embeddings, and stores them in the **Chroma vector database** inside the **embeddings/** directory.

---

## Running the Chatbot UI

Start the Streamlit interface:

```
streamlit run app.py
```

Open in browser:

```
http://localhost:8501
```

---

## Running the API Server

Start the FastAPI server:

```
uvicorn api.main:app --reload
```

API endpoint:

```
http://localhost:8000/chat
```

Example request:

```
{
  "query": "What services does the company provide?"
}
```

---

## Implemented Novelties

### 1. Retrieval-Augmented Generation with Local Vector Database

The chatbot retrieves relevant document chunks from a **Chroma vector database** before generating responses, ensuring answers are grounded in real data.

Reference files:

* chatbot.py
* rag_build.py

---

### 2. Multi-layer Hallucination Guardrails

To improve reliability, the chatbot includes multiple validation layers:

* Context sufficiency checking
* Query–context keyword overlap verification
* Strict prompt constraints
* Post-answer grounding validation

If the system cannot confidently answer, it returns a **safe refusal response**.

Reference file:

* chatbot.py

---

### 3. Sentiment Analysis on Assistant Responses

Each generated response is analyzed using a sentiment classifier. The sentiment score is displayed in the user interface.

Reference files:

* sentiment.py
* app.py

---

### 4. User-Controlled Retrieval and Model Selection

Users can dynamically configure:

* Number of retrieved documents (Top-K)
* LLM model used by Ollama

This allows flexible experimentation and optimization.

Reference file:

* app.py

---

### 5. Dual Interface: UI and API

The chatbot can be accessed through:

* **Streamlit Web Interface** for interactive chatting
* **FastAPI Endpoint** for programmatic integration

Reference files:

* app.py
* api/main.py

---

### 6. Source Citation and Grounding Confidence

Each answer includes:

* Source document snippets
* Similarity scores
* Grounding confidence level

This improves transparency and allows users to verify the origin of the information.

Reference files:

* chatbot.py
* app.py

---

## Example Chat Flow

User Question:

```
What services does the company provide?
```

Processing Steps:

1. Query is converted into embeddings
2. Similar document chunks are retrieved
3. Context is provided to the LLM
4. Guardrails validate the generated response
5. Sentiment analysis is performed
6. Final response is returned with confidence score and sources

---

## Technologies Used

Python – Backend development
Streamlit – Web interface
FastAPI – REST API
Ollama – Local LLM inference
ChromaDB – Vector database
Sentence Transformers – Text embeddings
LangChain – RAG pipeline
Transformers – Sentiment analysis

---

## Future Improvements

* Conversation memory
* File upload support in UI
* Authentication and user management
* Deployment with Docker
* Evaluation metrics for answer quality

---

## Author

OMPRAKASH S

Project: Intelligent Business RAG Chatbot

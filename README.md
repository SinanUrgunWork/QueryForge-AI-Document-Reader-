# queryforge

Production-ready Retrieval-Augmented Generation (RAG) API.
Upload documents, ask questions, get grounded answers with source citations and quality scores.

---

## What it does

```
User uploads PDF / TXT
        |
        v
Document is split into chunks
        |
        v
Each chunk is embedded (converted to a vector)
        |
        v
Vectors stored in FAISS index
        |
        v
User asks a question
        |
        v
Question embedded, top-k similar chunks retrieved
        |
        v
LLM generates answer grounded in retrieved chunks
        |
        v
Response returned with answer + sources + quality scores
```

---

## Tech stack

| Layer | Tool | Version |
|---|---|---|
| API framework | FastAPI | 0.115+ |
| LLM orchestration | LangChain | 0.3+ |
| Embeddings | OpenAI text-embedding-3-small | latest |
| Vector store | FAISS | 1.8+ |
| LLM | OpenAI GPT-4o-mini (default) | latest |
| Document parsing | pypdf, python-docx | latest |
| Containerization | Docker + Docker Compose | latest |
| Monitoring | Prometheus metrics endpoint | latest |

---

## Project structure

```
queryforge/
    src/
        api/
            __init__.py
            routes.py          FastAPI route handlers
            schemas.py         Pydantic request/response models
        ingestion/
            __init__.py
            loader.py          Load PDF, TXT, DOCX files
            splitter.py        Chunk documents
            embedder.py        Embed chunks, add to FAISS
        retrieval/
            __init__.py
            store.py           FAISS index wrapper (save, load, search)
        generation/
            __init__.py
            chain.py           LangChain RAG chain
            prompts.py         Prompt templates
        evaluation/
            __init__.py
            scorer.py          Faithfulness + relevance scoring
        monitoring/
            __init__.py
            metrics.py         Prometheus counters and histograms
        config.py              All settings from env
        main.py                FastAPI app entry point
    tests/
        test_ingestion.py
        test_retrieval.py
        test_generation.py
    docker/
        Dockerfile
        docker-compose.yml
    .env.example
    requirements.txt
    config.yaml
```

---

## Step-by-step build guide

### Step 1 — Project setup

```bash
mkdir queryforge && cd queryforge
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install fastapi uvicorn langchain langchain-openai faiss-cpu \
            pypdf python-docx pydantic prometheus-client python-dotenv \
            pyyaml pytest httpx
pip freeze > requirements.txt
```

Create `.env`:

```
OPENAI_API_KEY=your_key_here
EMBEDDING_MODEL=text-embedding-3-small
LLM_MODEL=gpt-4o-mini
CHUNK_SIZE=512
CHUNK_OVERLAP=64
TOP_K=5
SCORE_THRESHOLD=0.0
MAX_TOKENS=1024
TEMPERATURE=0.1
FAISS_INDEX_PATH=./data/faiss_index
```

---

### Step 2 — Config (src/config.py)

```python
import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 512))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 64))
TOP_K = int(os.getenv("TOP_K", 5))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", 1024))
TEMPERATURE = float(os.getenv("TEMPERATURE", 0.1))
FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "./data/faiss_index")
```

---

### Step 3 — Document loading (src/ingestion/loader.py)

```python
from pathlib import Path
from pypdf import PdfReader
import docx


def load_file(path: str) -> str:
    suffix = Path(path).suffix.lower()

    if suffix == ".pdf":
        reader = PdfReader(path)
        return "\n".join(page.extract_text() or "" for page in reader.pages)

    if suffix == ".txt":
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    if suffix == ".docx":
        doc = docx.Document(path)
        return "\n".join(para.text for para in doc.paragraphs)

    raise ValueError(f"Unsupported file type: {suffix}")
```

---

### Step 4 — Text splitting (src/ingestion/splitter.py)

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter
from src.config import CHUNK_SIZE, CHUNK_OVERLAP


def split_text(text: str, source: str) -> list[dict]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " "],
    )
    chunks = splitter.split_text(text)
    return [{"content": chunk, "source": source, "chunk_id": i}
            for i, chunk in enumerate(chunks)]
```

---

### Step 5 — Vector store (src/retrieval/store.py)

```python
import os
import json
import faiss
import numpy as np
from openai import OpenAI
from src.config import OPENAI_API_KEY, EMBEDDING_MODEL, FAISS_INDEX_PATH

client = OpenAI(api_key=OPENAI_API_KEY)
_index = None
_metadata = []


def _embed(texts: list[str]) -> np.ndarray:
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    vectors = [item.embedding for item in response.data]
    return np.array(vectors, dtype="float32")


def add_chunks(chunks: list[dict]) -> int:
    global _index, _metadata
    texts = [c["content"] for c in chunks]
    vectors = _embed(texts)

    if _index is None:
        _index = faiss.IndexFlatL2(vectors.shape[1])

    _index.add(vectors)
    _metadata.extend(chunks)
    _save()
    return len(chunks)


def search(query: str, top_k: int = 5) -> list[dict]:
    if _index is None or _index.ntotal == 0:
        return []

    vector = _embed([query])
    distances, indices = _index.search(vector, top_k)

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx == -1:
            continue
        chunk = _metadata[idx].copy()
        chunk["score"] = float(1 / (1 + dist))
        results.append(chunk)
    return results


def _save():
    os.makedirs(FAISS_INDEX_PATH, exist_ok=True)
    faiss.write_index(_index, f"{FAISS_INDEX_PATH}/index.faiss")
    with open(f"{FAISS_INDEX_PATH}/metadata.json", "w") as f:
        json.dump(_metadata, f)


def load_index():
    global _index, _metadata
    index_file = f"{FAISS_INDEX_PATH}/index.faiss"
    meta_file = f"{FAISS_INDEX_PATH}/metadata.json"
    if os.path.exists(index_file):
        _index = faiss.read_index(index_file)
        with open(meta_file) as f:
            _metadata = json.load(f)
```

---

### Step 6 — Prompt templates (src/generation/prompts.py)

```python
from langchain.prompts import PromptTemplate

RAG_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are a precise assistant. Answer the question using ONLY
the context below. If the answer is not in the context, say so clearly.

Context:
{context}

Question: {question}

Answer:"""
)
```

---

### Step 7 — RAG chain (src/generation/chain.py)

```python
from langchain_openai import ChatOpenAI
from src.generation.prompts import RAG_PROMPT
from src.retrieval.store import search
from src.config import LLM_MODEL, TEMPERATURE, MAX_TOKENS, TOP_K


def run(question: str, top_k: int = TOP_K) -> dict:
    chunks = search(question, top_k=top_k)

    if not chunks:
        return {
            "answer": "No relevant documents found.",
            "sources": [],
            "chunks_used": 0,
        }

    context = "\n\n".join(
        f"[Source: {c['source']}, chunk {c['chunk_id']}]\n{c['content']}"
        for c in chunks
    )

    llm = ChatOpenAI(model=LLM_MODEL, temperature=TEMPERATURE, max_tokens=MAX_TOKENS)
    prompt = RAG_PROMPT.format(context=context, question=question)
    response = llm.invoke(prompt)

    return {
        "answer": response.content,
        "sources": chunks,
        "chunks_used": len(chunks),
    }
```

---

### Step 8 — Evaluation scores (src/evaluation/scorer.py)

```python
def score_relevance(question: str, chunks: list[dict]) -> float:
    if not chunks:
        return 0.0
    scores = [c.get("score", 0.0) for c in chunks]
    return round(sum(scores) / len(scores), 3)


def score_faithfulness(answer: str, chunks: list[dict]) -> float:
    if not chunks or not answer:
        return 0.0
    context_words = set(
        " ".join(c["content"] for c in chunks).lower().split()
    )
    answer_words = answer.lower().split()
    if not answer_words:
        return 0.0
    overlap = sum(1 for w in answer_words if w in context_words)
    return round(overlap / len(answer_words), 3)
```

---

### Step 9 — Prometheus metrics (src/monitoring/metrics.py)

```python
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

ingest_counter = Counter("queryforge_ingestions_total", "Total documents ingested")
query_counter = Counter("queryforge_queries_total", "Total queries processed")
query_latency = Histogram("queryforge_query_seconds", "Query latency in seconds")
faithfulness_histogram = Histogram(
    "queryforge_faithfulness_score",
    "Faithfulness scores",
    buckets=[0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
)
```

---

### Step 10 — API schemas (src/api/schemas.py)

```python
from pydantic import BaseModel


class QueryRequest(BaseModel):
    question: str
    top_k: int = 5
    return_sources: bool = True


class SourceChunk(BaseModel):
    content: str
    source: str
    chunk_id: int
    score: float


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceChunk]
    chunks_used: int
    evaluation: dict


class IngestResponse(BaseModel):
    filename: str
    chunks_added: int
    status: str
```

---

### Step 11 — Routes (src/api/routes.py)

```python
import time
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import Response
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

from src.api.schemas import QueryRequest, QueryResponse, IngestResponse
from src.ingestion.loader import load_file
from src.ingestion.splitter import split_text
from src.retrieval.store import add_chunks
from src.generation.chain import run
from src.evaluation.scorer import score_relevance, score_faithfulness
from src.monitoring.metrics import (
    ingest_counter, query_counter, query_latency, faithfulness_histogram
)

import tempfile, os

router = APIRouter()


@router.post("/ingest", response_model=IngestResponse)
async def ingest(file: UploadFile = File(...)):
    allowed = {".pdf", ".txt", ".docx"}
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in allowed:
        raise HTTPException(400, f"Unsupported type: {ext}")

    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        text = load_file(tmp_path)
        chunks = split_text(text, source=file.filename)
        added = add_chunks(chunks)
        ingest_counter.inc()
        return IngestResponse(filename=file.filename, chunks_added=added, status="ok")
    finally:
        os.unlink(tmp_path)


@router.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    start = time.time()
    result = run(req.question, top_k=req.top_k)

    relevance = score_relevance(req.question, result["sources"])
    faithfulness = score_faithfulness(result["answer"], result["sources"])
    faithfulness_histogram.observe(faithfulness)

    query_counter.inc()
    query_latency.observe(time.time() - start)

    sources = result["sources"] if req.return_sources else []
    return QueryResponse(
        answer=result["answer"],
        sources=sources,
        chunks_used=result["chunks_used"],
        evaluation={"relevance": relevance, "faithfulness": faithfulness},
    )


@router.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@router.get("/health")
def health():
    return {"status": "ok"}
```

---

### Step 12 — App entry point (src/main.py)

```python
from fastapi import FastAPI
from src.api.routes import router
from src.retrieval.store import load_index

app = FastAPI(title="queryforge", version="1.0.0")
app.include_router(router)


@app.on_event("startup")
def startup():
    load_index()
```

---

### Step 13 — Docker

**docker/Dockerfile:**
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**docker/docker-compose.yml:**
```yaml
services:
  api:
    build:
      context: ..
      dockerfile: docker/Dockerfile
    ports:
      - "8000:8000"
    env_file:
      - .env
    volumes:
      - ./data:/app/data
```

---

## Running locally

```bash
# 1. Clone and install
git clone https://github.com/SinanUrgunWork/queryforge
cd queryforge
pip install -r requirements.txt

# 2. Set env variables
cp .env.example .env
# Fill in your OPENAI_API_KEY

# 3. Run
uvicorn src.main:app --reload

# API docs at http://localhost:8000/docs
```

---

## API usage examples

### Ingest a document

```bash
curl -X POST http://localhost:8000/ingest \
  -F "file=@document.pdf"
```

Response:
```json
{
  "filename": "document.pdf",
  "chunks_added": 42,
  "status": "ok"
}
```

### Ask a question

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the main findings?", "top_k": 5}'
```

Response:
```json
{
  "answer": "The main findings are...",
  "sources": [
    {
      "content": "...",
      "source": "document.pdf",
      "chunk_id": 3,
      "score": 0.87
    }
  ],
  "chunks_used": 5,
  "evaluation": {
    "relevance": 0.85,
    "faithfulness": 0.91
  }
}
```

---

## Endpoints

| Method | Path | Description |
|---|---|---|
| POST | /ingest | Upload and index a document |
| POST | /query | Ask a question |
| GET | /metrics | Prometheus metrics |
| GET | /health | Health check |
| GET | /docs | Auto-generated Swagger UI |

---

## Evaluation metrics

Every query response includes two scores:

**Faithfulness** — what fraction of the answer words appear in the retrieved context. High score means the answer is grounded, not hallucinated.

**Relevance** — average similarity score of retrieved chunks to the question. High score means retrieval found the right parts.

---

## Build order summary

1. `src/config.py` — env variables
2. `src/ingestion/loader.py` — file loading
3. `src/ingestion/splitter.py` — chunking
4. `src/retrieval/store.py` — FAISS index
5. `src/generation/prompts.py` — prompt template
6. `src/generation/chain.py` — RAG chain
7. `src/evaluation/scorer.py` — quality scores
8. `src/monitoring/metrics.py` — Prometheus
9. `src/api/schemas.py` — Pydantic models
10. `src/api/routes.py` — endpoints
11. `src/main.py` — app startup
12. `docker/` — containerize

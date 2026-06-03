# QueryForge

**Production-ready Retrieval-Augmented Generation (RAG) API**

Upload documents, ask questions, get grounded answers with source citations and quality scores — powered by OpenAI and FAISS.

---

## Overview

QueryForge lets you turn any collection of PDF, TXT, or DOCX files into a queryable knowledge base. Documents are split into chunks, embedded into vectors, and stored in a FAISS index. When you ask a question, the most relevant chunks are retrieved and passed to an LLM to generate a grounded answer — with faithfulness and relevance scores included in every response.

```
Upload document (PDF / TXT / DOCX)
        ↓
Split into chunks → Embed → Store in FAISS
        ↓
Ask a question
        ↓
Retrieve top-k similar chunks
        ↓
LLM generates grounded answer
        ↓
Return answer + sources + quality scores
```

---

## Features

- **Multi-format ingestion** — PDF, TXT, DOCX support out of the box
- **Semantic search** — OpenAI embeddings + FAISS vector index
- **Grounded answers** — LLM answers using only retrieved context, reducing hallucination
- **Quality scores** — Faithfulness and relevance metrics on every query
- **Prometheus metrics** — `/metrics` endpoint for observability
- **Swagger UI** — Auto-generated interactive docs at `/docs`
- **Docker ready** — Single `docker compose up` to run

---

## Tech Stack

| Layer | Tool | Version |
|---|---|---|
| API framework | FastAPI | 0.115+ |
| LLM orchestration | LangChain | 0.3+ |
| Embeddings | OpenAI text-embedding-3-small | latest |
| Vector store | FAISS | 1.8+ |
| LLM | OpenAI GPT-4o-mini | latest |
| Document parsing | pypdf, python-docx | latest |
| Containerization | Docker + Docker Compose | latest |
| Monitoring | Prometheus | latest |

---

## Project Structure

```
queryforge/
├── src/
│   ├── api/
│   │   ├── routes.py          # FastAPI route handlers
│   │   └── schemas.py         # Pydantic request/response models
│   ├── ingestion/
│   │   ├── loader.py          # PDF, TXT, DOCX loading
│   │   └── splitter.py        # Recursive text chunking
│   ├── retrieval/
│   │   └── store.py           # FAISS index (add, search, persist)
│   ├── generation/
│   │   ├── chain.py           # LangChain RAG chain
│   │   └── prompts.py         # Prompt templates
│   ├── evaluation/
│   │   └── scorer.py          # Faithfulness + relevance scoring
│   ├── monitoring/
│   │   └── metrics.py         # Prometheus counters and histograms
│   ├── config.py              # Environment-based settings
│   └── main.py                # FastAPI app entry point
├── tests/
│   ├── test_ingestion.py
│   ├── test_retrieval.py
│   └── test_generation.py
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── .env.example
├── config.yaml
└── requirements.txt
```

---

## Getting Started

### 1. Clone and install

```bash
git clone https://github.com/SinanUrgunWork/queryforge
cd queryforge
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS / Linux
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
```

Open `.env` and set your OpenAI API key:

```
OPENAI_API_KEY=sk-...
```

### 3. Run

```bash
uvicorn src.main:app --reload
```

API is live at `http://localhost:8000`
Swagger UI at `http://localhost:8000/docs`

---

## Docker

```bash
cp .env.example .env
# fill in OPENAI_API_KEY in .env

docker compose -f docker/docker-compose.yml up --build
```

---

## API Reference

### POST `/ingest`

Upload a document to the knowledge base.

```bash
curl -X POST http://localhost:8000/ingest \
  -F "file=@document.pdf"
```

```json
{
  "filename": "document.pdf",
  "chunks_added": 42,
  "status": "ok"
}
```

---

### POST `/query`

Ask a question against the indexed documents.

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the main findings?", "top_k": 5}'
```

```json
{
  "answer": "The main findings are ...",
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

**Request body:**

| Field | Type | Default | Description |
|---|---|---|---|
| `question` | string | required | The question to answer |
| `top_k` | int | 5 | Number of chunks to retrieve |
| `return_sources` | bool | true | Include source chunks in response |

---

### GET `/health`

```json
{ "status": "ok" }
```

### GET `/metrics`

Prometheus metrics endpoint — scrape with any Prometheus-compatible collector.

### GET `/docs`

Auto-generated Swagger UI.

---

## Evaluation Scores

Every `/query` response includes two scores:

**Faithfulness** — fraction of answer words that appear in the retrieved context. A high score means the answer is grounded in the documents, not hallucinated.

**Relevance** — average similarity score of retrieved chunks to the question. A high score means retrieval found the right parts of the document.

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | — | Required. Your OpenAI API key |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | OpenAI embedding model |
| `LLM_MODEL` | `gpt-4o-mini` | OpenAI chat model |
| `CHUNK_SIZE` | `512` | Max characters per chunk |
| `CHUNK_OVERLAP` | `64` | Overlap between consecutive chunks |
| `TOP_K` | `5` | Default number of chunks to retrieve |
| `MAX_TOKENS` | `1024` | Max tokens in LLM response |
| `TEMPERATURE` | `0.1` | LLM sampling temperature |
| `FAISS_INDEX_PATH` | `./data/faiss_index` | Path to persist FAISS index |

---

## Running Tests

```bash
pytest tests/
```

---

## License

MIT

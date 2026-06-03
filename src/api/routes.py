import time
import tempfile
import os
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

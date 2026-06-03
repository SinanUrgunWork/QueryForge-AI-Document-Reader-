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

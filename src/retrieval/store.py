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

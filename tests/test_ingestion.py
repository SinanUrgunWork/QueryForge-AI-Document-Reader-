import pytest
from src.ingestion.splitter import split_text


def test_split_text_basic():
    text = "Hello world. " * 100
    chunks = split_text(text, source="test.txt")
    assert len(chunks) > 0
    assert all("content" in c for c in chunks)
    assert all("source" in c for c in chunks)
    assert all(c["source"] == "test.txt" for c in chunks)


def test_split_text_chunk_ids():
    text = "paragraph one\n\nparagraph two\n\nparagraph three"
    chunks = split_text(text, source="test.txt")
    ids = [c["chunk_id"] for c in chunks]
    assert ids == list(range(len(chunks)))

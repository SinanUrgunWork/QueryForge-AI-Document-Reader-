import pytest
from unittest.mock import patch, MagicMock
import numpy as np


def test_search_empty_index():
    from src.retrieval import store
    store._index = None
    store._metadata = []
    results = store.search("any question")
    assert results == []

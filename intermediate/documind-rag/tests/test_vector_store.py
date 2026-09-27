"""
test_vector_store.py

Unit tests for VectorStore in vector_store.py.

Synthetic numpy vectors are used directly, so these tests do NOT require
the sentence-transformers model to be downloaded, need no API key, and
make NO network calls.
"""

import sys
import os

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from vector_store import VectorStore


def _make_chunk(text, source="doc.txt", chunk_id=0, page=None):
    return {"text": text, "source": source, "chunk_id": chunk_id, "page": page}


def _normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (vectors / norms).astype("float32")


def test_empty_store_is_empty():
    store = VectorStore()
    assert store.is_empty() is True
    assert store.chunk_count() == 0


def test_empty_store_search_returns_no_results():
    store = VectorStore()
    query = _normalize(np.array([[1.0, 0.0, 0.0]], dtype="float32"))
    results = store.search(query, top_k=3)
    assert results == []


def test_vectors_can_be_added():
    store = VectorStore()
    vectors = _normalize(np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype="float32"))
    chunks = [_make_chunk("chunk A", chunk_id=0), _make_chunk("chunk B", chunk_id=1)]
    store.add(vectors, chunks)

    assert store.is_empty() is False
    assert store.chunk_count() == 2


def test_search_returns_results():
    store = VectorStore()
    vectors = _normalize(
        np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype="float32")
    )
    chunks = [
        _make_chunk("about cats", chunk_id=0),
        _make_chunk("about dogs", chunk_id=1),
        _make_chunk("about birds", chunk_id=2),
    ]
    store.add(vectors, chunks)

    query = _normalize(np.array([[1.0, 0.0, 0.0]], dtype="float32"))
    results = store.search(query, top_k=1)

    assert len(results) == 1
    assert results[0]["text"] == "about cats"
    assert "score" in results[0]


def test_top_k_limits_number_of_results():
    store = VectorStore()
    vectors = _normalize(np.random.rand(10, 5).astype("float32"))
    chunks = [_make_chunk(f"chunk {i}", chunk_id=i) for i in range(10)]
    store.add(vectors, chunks)

    query = _normalize(np.random.rand(1, 5).astype("float32"))
    results = store.search(query, top_k=3)

    assert len(results) == 3


def test_top_k_capped_by_available_chunks():
    store = VectorStore()
    vectors = _normalize(np.array([[1.0, 0.0], [0.0, 1.0]], dtype="float32"))
    chunks = [_make_chunk("A", chunk_id=0), _make_chunk("B", chunk_id=1)]
    store.add(vectors, chunks)

    query = _normalize(np.array([[1.0, 0.0]], dtype="float32"))
    results = store.search(query, top_k=10)

    assert len(results) == 2


def test_clear_resets_store():
    store = VectorStore()
    vectors = _normalize(np.array([[1.0, 0.0]], dtype="float32"))
    chunks = [_make_chunk("A", chunk_id=0)]
    store.add(vectors, chunks)
    assert store.is_empty() is False

    store.clear()
    assert store.is_empty() is True
    assert store.chunk_count() == 0


def test_source_count_and_names():
    store = VectorStore()
    vectors = _normalize(np.array([[1.0, 0.0], [0.0, 1.0]], dtype="float32"))
    chunks = [
        _make_chunk("A", source="doc1.txt", chunk_id=0),
        _make_chunk("B", source="doc2.txt", chunk_id=0),
    ]
    store.add(vectors, chunks)

    assert store.source_count() == 2
    assert set(store.source_names()) == {"doc1.txt", "doc2.txt"}

"""
vector_store.py

Stage 4 of the RAG pipeline: vector indexing and similarity search.

An in-memory FAISS index (IndexFlatIP over L2-normalized vectors, which
is equivalent to cosine-similarity search) is used. Persistence to disk
is intentionally NOT implemented -- this is a deliberate design choice
for this internship/demo project:

  - Documents are uploaded fresh in each session, so there is no need to
    keep a durable index between app restarts.
  - Avoiding on-disk persistence keeps the project simple, keeps no user
    document content on disk, and avoids extra dependencies/config for
    a demo-scale application.
  - See the README "Design Decisions" section for more detail.
"""

from typing import List, Dict, Any

import numpy as np
import faiss


class VectorStore:
    """A simple in-memory FAISS-backed vector store with metadata."""

    def __init__(self, embedding_dim: int = None):
        self._embedding_dim = embedding_dim
        self._index = None  # created lazily once we know the embedding dim
        self._metadata: List[Dict[str, Any]] = []  # parallel to index rows

    def _ensure_index(self, dim: int) -> None:
        if self._index is None:
            self._embedding_dim = dim
            self._index = faiss.IndexFlatIP(dim)

    def add(self, vectors: np.ndarray, chunks: List[Dict[str, Any]]) -> None:
        """
        Add a batch of embedding vectors and their corresponding chunk
        metadata dicts to the store.

        `vectors` must be a float32 array of shape (n, dim), L2-normalized.
        `chunks` must have the same length as `vectors`, each containing
        at least: text, source, chunk_id, page.
        """
        if vectors is None or len(vectors) == 0:
            return
        if len(vectors) != len(chunks):
            raise ValueError("vectors and chunks must have the same length")

        self._ensure_index(vectors.shape[1])
        self._index.add(vectors)
        self._metadata.extend(chunks)

    def search(self, query_vector: np.ndarray, top_k: int = 4) -> List[Dict[str, Any]]:
        """
        Search for the top_k most similar chunks to the given query
        vector (float32, shape (1, dim), L2-normalized).

        Returns a list of result dicts:
            {
                "text": str,
                "source": str,
                "chunk_id": int,
                "page": int or None,
                "score": float,  # cosine similarity, higher is more relevant
            }
        Sorted by descending similarity score.
        """
        if self.is_empty():
            return []

        k = min(top_k, len(self._metadata))
        scores, indices = self._index.search(query_vector, k)

        results: List[Dict[str, Any]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            meta = self._metadata[idx]
            results.append(
                {
                    "text": meta["text"],
                    "source": meta["source"],
                    "chunk_id": meta["chunk_id"],
                    "page": meta.get("page"),
                    "score": float(score),
                }
            )
        return results

    def clear(self) -> None:
        """Remove all vectors and metadata from the store."""
        self._index = None
        self._embedding_dim = None
        self._metadata = []

    def is_empty(self) -> bool:
        """Return True if the store has no vectors indexed."""
        return self._index is None or self._index.ntotal == 0

    def chunk_count(self) -> int:
        """Return the number of chunks currently indexed."""
        return len(self._metadata)

    def source_count(self) -> int:
        """Return the number of distinct source documents indexed."""
        return len({m["source"] for m in self._metadata})

    def source_names(self) -> List[str]:
        """Return the list of distinct source document filenames, in order seen."""
        seen = []
        for m in self._metadata:
            if m["source"] not in seen:
                seen.append(m["source"])
        return seen

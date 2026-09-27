"""
embeddings.py

Stage 3 of the RAG pipeline: embedding generation.

Embeddings are generated entirely locally using sentence-transformers
(all-MiniLM-L6-v2 by default), so there is no embedding API cost and no
dependency on an external provider for this stage of the pipeline. This
keeps retrieval fully independent from the LLM generation provider
(OpenRouter), which is only used for the final answer-generation step.

The model is loaded once and cached (functools.lru_cache), so it is not
reloaded on every query.
"""

from functools import lru_cache
from typing import List, Dict, Any

import numpy as np
from sentence_transformers import SentenceTransformer

from config import EMBEDDING_MODEL_NAME


@lru_cache(maxsize=1)
def load_embedding_model() -> SentenceTransformer:
    """
    Load (and cache) the local sentence-transformers embedding model.

    Using lru_cache ensures the (relatively expensive) model load only
    happens once per process, regardless of how many times this function
    is called.
    """
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


def embed_documents(chunks: List[Dict[str, Any]]) -> np.ndarray:
    """
    Embed a list of chunk dicts (each containing a "text" key).

    Returns a float32 numpy array of shape (num_chunks, embedding_dim),
    L2-normalized so that inner-product search in FAISS behaves as
    cosine-similarity search.
    """
    if not chunks:
        return np.zeros((0, 0), dtype="float32")

    model = load_embedding_model()
    texts = [c["text"] for c in chunks]

    vectors = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return vectors.astype("float32")


def embed_query(query: str) -> np.ndarray:
    """
    Embed a single query string.

    Returns a float32 numpy array of shape (1, embedding_dim),
    L2-normalized to match the document embedding space.
    """
    model = load_embedding_model()
    vector = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return vector.astype("float32")

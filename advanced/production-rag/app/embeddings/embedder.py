from functools import lru_cache
import numpy as np

class Embedder:
    """Lazy, process-wide local embedding model wrapper."""
    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model = None

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype="float32")
        vectors = self._get_model().encode(texts, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
        return np.asarray(vectors, dtype="float32")

@lru_cache(maxsize=2)
def get_embedder(model_name: str) -> Embedder:
    return Embedder(model_name)

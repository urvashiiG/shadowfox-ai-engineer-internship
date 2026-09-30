import numpy as np
import pytest
from app.config.settings import Settings
from app.services.document_service import DocumentService

class DeterministicEmbedder:
    def encode(self, texts):
        rows = []
        for text in texts:
            lower = text.lower()
            rows.append([float(any(word in lower for word in group)) for group in
                         [("paris", "france", "eiffel"), ("1889", "completed", "built"), ("iron", "metal"), ("population", "people")]])
        matrix = np.array(rows, dtype="float32")
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        return matrix / np.maximum(norms, 1e-9)

@pytest.fixture
def service(tmp_path):
    from app.vectorstore.faiss_store import FaissStore
    settings = Settings(_env_file=None, data_dir=tmp_path, chunk_size=100, chunk_overlap=10, top_k=8, rerank_top_k=4)
    embedder = DeterministicEmbedder()
    return DocumentService(settings, FaissStore(tmp_path / "indexes", embedder))

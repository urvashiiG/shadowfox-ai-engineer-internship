import json
from pathlib import Path
from threading import RLock
import numpy as np

class FaissStore:
    """Local cosine-search store; normalized vectors use an inner-product index."""
    def __init__(self, directory: Path, embedder):
        self.directory, self.embedder = Path(directory), embedder
        self.directory.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()
        self._faiss = None
        self._vectors = np.empty((0, 0), dtype="float32")
        self._metadata: list[dict] = []
        self._load()

    def _load(self):
        metadata_path, vectors_path = self.directory / "metadata.json", self.directory / "vectors.npy"
        if metadata_path.exists() and vectors_path.exists():
            self._metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            self._vectors = np.load(vectors_path, allow_pickle=False).astype("float32")
            index_path = self.directory / "index.faiss"
            if index_path.exists():
                import faiss
                self._faiss = faiss.read_index(str(index_path))
                if self._faiss.ntotal != len(self._metadata):
                    self._rebuild_index()
            else:
                self._rebuild_index()

    def _rebuild_index(self):
        if not len(self._metadata):
            self._faiss = None
            return
        import faiss
        self._faiss = faiss.IndexFlatIP(self._vectors.shape[1])
        self._faiss.add(np.ascontiguousarray(self._vectors))

    def _persist(self):
        self.directory.mkdir(parents=True, exist_ok=True)
        np.save(self.directory / "vectors.npy", self._vectors, allow_pickle=False)
        (self.directory / "metadata.json").write_text(json.dumps(self._metadata, ensure_ascii=False), encoding="utf-8")
        if self._faiss is not None:
            import faiss
            faiss.write_index(self._faiss, str(self.directory / "index.faiss"))
        else:
            (self.directory / "index.faiss").unlink(missing_ok=True)

    def add(self, chunks: list[dict]) -> None:
        if not chunks:
            return
        with self._lock:
            ids = {item["document_id"] for item in self._metadata}
            if any(chunk["document_id"] in ids for chunk in chunks):
                raise ValueError("This document is already indexed.")
            vectors = self.embedder.encode([chunk["source_text"] for chunk in chunks])
            if vectors.ndim != 2 or vectors.shape[0] != len(chunks):
                raise ValueError("The embedding model returned invalid vectors.")
            if len(self._metadata) and vectors.shape[1] != self._vectors.shape[1]:
                raise ValueError("Embedding dimensions do not match the existing index.")
            self._vectors = vectors if not len(self._metadata) else np.vstack((self._vectors, vectors))
            self._metadata.extend(chunks)
            self._rebuild_index()
            self._persist()

    def search(self, query: str, top_k: int = 8, document_id: str | None = None) -> list[dict]:
        with self._lock:
            if not self._metadata:
                return []
            vector = self.embedder.encode([query])
            if self._faiss is None:
                return []
            scores, indices = self._faiss.search(np.ascontiguousarray(vector), len(self._metadata))
            found = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < 0:
                    continue
                meta = self._metadata[int(idx)]
                if document_id is None or meta["document_id"] == document_id:
                    found.append({**meta, "semantic_score": float(score), "score": float(score)})
                    if len(found) == top_k:
                        break
            return found

    def remove_document(self, document_id: str) -> int:
        with self._lock:
            keep = [i for i, item in enumerate(self._metadata) if item["document_id"] != document_id]
            removed = len(self._metadata) - len(keep)
            if removed:
                self._metadata = [self._metadata[i] for i in keep]
                self._vectors = self._vectors[keep] if keep else np.empty((0, self._vectors.shape[1]), dtype="float32")
                self._rebuild_index()
                self._persist()
            return removed

    def documents(self) -> list[dict]:
        by_id = {}
        for row in self._metadata:
            item = by_id.setdefault(row["document_id"], {"document_id": row["document_id"], "document_name": row["document_name"], "source_type": row["source_type"], "chunk_count": 0})
            item["chunk_count"] += 1
        return list(by_id.values())

    def count(self) -> int:
        return len(self._metadata)

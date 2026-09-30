from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4
from app.config.settings import get_settings
from app.embeddings.embedder import get_embedder
from app.ingestion.loader import load_document
from app.processing.chunker import chunk_pages
from app.vectorstore.faiss_store import FaissStore

class DocumentService:
    def __init__(self, settings=None, store=None):
        self.settings = settings or get_settings()
        self.embedder = get_embedder(self.settings.embedding_model)
        self.store = store or FaissStore(self.settings.index_dir, self.embedder)
        self._documents: dict[str, dict] = {}
        for doc in self.store.documents():
            doc["uploaded_at"] = "Previously indexed"
            self._documents[doc["document_id"]] = doc

    def upload(self, file, filename: str) -> dict:
        pages = load_document(file, filename, self.settings.max_upload_bytes)
        document_id = uuid4().hex
        chunks = chunk_pages(pages, document_id=document_id, document_name=Path(filename).name,
                             source_type=Path(filename).suffix.lower().lstrip("."),
                             size=self.settings.chunk_size, overlap=self.settings.chunk_overlap)
        if not chunks:
            raise ValueError("This file did not produce any usable text chunks.")
        self.store.add(chunks)
        doc = {"document_id": document_id, "document_name": Path(filename).name,
               "source_type": Path(filename).suffix.lower().lstrip("."), "chunk_count": len(chunks),
               "uploaded_at": datetime.now(timezone.utc).isoformat()}
        self._documents[document_id] = doc
        return doc

    def list_documents(self) -> list[dict]:
        return list(self._documents.values())

    def delete(self, document_id: str) -> bool:
        if document_id not in self._documents:
            return False
        self.store.remove_document(document_id)
        self._documents.pop(document_id, None)
        return True

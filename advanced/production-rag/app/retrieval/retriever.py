from app.vectorstore.faiss_store import FaissStore

def retrieve_candidates(store: FaissStore, query: str, candidate_top_k: int, document_id: str | None = None) -> list[dict]:
    return store.search(query, candidate_top_k, document_id)

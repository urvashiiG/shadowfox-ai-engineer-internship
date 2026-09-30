from typing import TypedDict, Any

class RAGState(TypedDict, total=False):
    original_query: str
    rewritten_query: str
    document_id: str | None
    retrieved_chunks: list[dict[str, Any]]
    reranked_chunks: list[dict[str, Any]]
    context: str
    sources: list[dict[str, Any]]
    answer: str
    grounded: bool
    error: str | None
    candidate_count: int

from pydantic import BaseModel
from typing import Any

class QueryResponse(BaseModel):
    answer: str
    grounded: bool
    grounding_status: str
    rewritten_query: str
    sources: list[dict[str, Any]]
    candidate_count: int
    context_count: int
    error: str | None = None

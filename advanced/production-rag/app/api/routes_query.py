from fastapi import APIRouter, Depends, HTTPException
from app.generation.llm import OpenRouterLLM
from app.schemas.queries import QueryRequest
from app.schemas.responses import QueryResponse
from app.services.document_service import DocumentService
from app.workflow.graph import run_query

router = APIRouter(tags=["query"])

def get_service() -> DocumentService:
    from app.main import document_service
    return document_service

@router.post("/query", response_model=QueryResponse)
def query_documents(request: QueryRequest, service: DocumentService = Depends(get_service)):
    if request.document_id and not any(d["document_id"] == request.document_id for d in service.list_documents()):
        raise HTTPException(status_code=404, detail="Selected document was not found.")
    try:
        return run_query(service.store, OpenRouterLLM(), request.question, request.document_id)
    except Exception as exc:
        raise HTTPException(status_code=503, detail="The RAG workflow is temporarily unavailable.") from exc

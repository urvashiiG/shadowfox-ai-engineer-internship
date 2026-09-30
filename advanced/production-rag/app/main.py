from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api import routes_documents, routes_health, routes_query
from app.services.document_service import DocumentService

document_service = DocumentService()
app = FastAPI(title="DocuMind AI API", version="1.0.0", description="Local-indexed, source-grounded document question answering")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.include_router(routes_health.router)
app.include_router(routes_documents.router)
app.include_router(routes_query.router)

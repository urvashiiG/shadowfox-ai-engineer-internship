from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from app.schemas.documents import DocumentInfo, UploadResponse
from app.services.document_service import DocumentService

router = APIRouter(prefix="/documents", tags=["documents"])

def get_service() -> DocumentService:
    from app.main import document_service
    return document_service

@router.post("/upload", response_model=UploadResponse, status_code=201)
def upload_document(file: UploadFile = File(...), service: DocumentService = Depends(get_service)):
    try:
        doc = service.upload(file.file, file.filename or "")
        return {"document": doc, "message": "Document indexed successfully."}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Document processing failed. Please try again.") from exc

@router.get("", response_model=list[DocumentInfo])
def list_documents(service: DocumentService = Depends(get_service)):
    return service.list_documents()

@router.delete("/{document_id}")
def delete_document(document_id: str, service: DocumentService = Depends(get_service)):
    if not service.delete(document_id):
        raise HTTPException(status_code=404, detail="Document not found.")
    return {"message": "Document deleted."}

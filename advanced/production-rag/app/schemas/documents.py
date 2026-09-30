from pydantic import BaseModel

class DocumentInfo(BaseModel):
    document_id: str
    document_name: str
    source_type: str
    chunk_count: int
    uploaded_at: str

class UploadResponse(BaseModel):
    document: DocumentInfo
    message: str

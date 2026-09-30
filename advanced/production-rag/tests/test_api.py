from fastapi.testclient import TestClient
from app.main import app
from app.api.routes_documents import get_service as docs_service
from app.api.routes_query import get_service as query_service

def test_health_and_api_validation(service):
    app.dependency_overrides[docs_service] = lambda: service
    app.dependency_overrides[query_service] = lambda: service
    client = TestClient(app)
    assert client.get("/health").status_code == 200
    assert client.post("/query", json={"question": "  "}).status_code == 422
    assert client.post("/query", json={"question": "What?", "document_id": "missing"}).status_code == 404
    assert client.post("/documents/upload", files={"file": ("bad.exe", b"x")}).status_code == 400
    app.dependency_overrides.clear()

def test_upload_documents_and_list(service):
    app.dependency_overrides[docs_service] = lambda: service
    client = TestClient(app)
    response = client.post("/documents/upload", files={"file": ("eiffel.md", b"Eiffel Tower in Paris", "text/markdown")})
    assert response.status_code == 201
    assert response.json()["document"]["source_type"] == "md"
    assert client.get("/documents").json()[0]["document_name"] == "eiffel.md"
    app.dependency_overrides.clear()

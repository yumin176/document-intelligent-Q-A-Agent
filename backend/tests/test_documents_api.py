from fastapi.testclient import TestClient

from app.dependencies import get_ingestion_service, get_rag_service
from app.ingestion.service import IngestionService
from app.main import create_app
from app.embedding.fake import FakeEmbedder

from app.vectorstore.memory import InMemoryVectorStore
def make_test_ingestion_service():
    return IngestionService(
        embedder=FakeEmbedder(),
        vector_store=InMemoryVectorStore()
    )

def test_upload_document_endpoint():
    app = create_app()
    app.dependency_overrides[get_ingestion_service] = make_test_ingestion_service
    client = TestClient(app)

    files = {"file": ("test.md", b"# title\ncontent", "text/markdown")}
    response = client.post("/api/documents", files=files)
    assert response.status_code == 200
    assert response.json()["chunk_count"] >= 1

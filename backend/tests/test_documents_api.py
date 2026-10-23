from fastapi.testclient import TestClient

from app.dependencies import get_bm25_refresher, get_ingestion_service
from app.embedding.fake import FakeEmbedder
from app.ingestion.service import IngestionService
from app.main import create_app
from app.vectorstore.memory import InMemoryVectorStore


def make_test_ingestion_service() -> IngestionService:
    return IngestionService(
        embedder=FakeEmbedder(dim=8),
        vector_store=InMemoryVectorStore(),
    )


def test_upload_document_endpoint() -> None:
    app = create_app()
    app.dependency_overrides[get_ingestion_service] = make_test_ingestion_service
    app.dependency_overrides[get_bm25_refresher] = lambda: (lambda: None)
    client = TestClient(app)

    files = {"file": ("test.md", b"# title\ncontent", "text/markdown")}
    response = client.post("/api/documents", files=files)

    assert response.status_code == 200
    body = response.json()
    assert body["chunk_count"] >= 1
    assert body["filename"] == "test.md"
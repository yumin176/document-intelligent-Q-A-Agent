from fastapi.testclient import TestClient

from app.dependencies import get_vector_store
from app.main import create_app
from app.vectorstore.base import ChunkRecord
from app.vectorstore.memory import InMemoryVectorStore


def make_store() -> InMemoryVectorStore:
    store = InMemoryVectorStore()
    store.upsert(
        [
            ChunkRecord(
                id="doc1#0:aaaaaaaa",
                text="第一段",
                embedding=[1.0, 0.0],
                metadata={
                    "document_id": "doc1",
                    "doc_name": "docs/one.md",
                    "filename": "one.md",
                    "file_type": "md",
                    "chunk_index": 0,
                    "chunk_hash": "hash1",
                    "chunk_strategy": "fixed",
                    "chunk_size": 500,
                    "chunk_overlap": 50,
                    "embedding_model": "fake",
                    "embedding_dimension": 2,
                },
            ),
            ChunkRecord(
                id="doc1#1:bbbbbbbb",
                text="第二段",
                embedding=[0.0, 1.0],
                metadata={
                    "document_id": "doc1",
                    "doc_name": "docs/one.md",
                    "filename": "one.md",
                    "file_type": "md",
                    "chunk_index": 1,
                    "chunk_hash": "hash2",
                    "chunk_strategy": "fixed",
                    "chunk_size": 500,
                    "chunk_overlap": 50,
                    "embedding_model": "fake",
                    "embedding_dimension": 2,
                },
            ),
            ChunkRecord(
                id="doc2#0:cccccccc",
                text="另一文档",
                embedding=[1.0, 1.0],
                metadata={
                    "document_id": "doc2",
                    "doc_name": "docs/two.md",
                    "filename": "two.md",
                    "file_type": "md",
                    "chunk_index": 0,
                    "chunk_hash": "hash3",
                    "chunk_strategy": "recursive",
                    "chunk_size": 300,
                    "chunk_overlap": 30,
                    "embedding_model": "fake",
                    "embedding_dimension": 2,
                },
            ),
        ]
    )
    return store


def make_client() -> TestClient:
    app = create_app()
    store = make_store()
    app.dependency_overrides[get_vector_store] = lambda: store
    return TestClient(app)


def test_knowledge_documents_are_aggregated() -> None:
    client = make_client()

    response = client.get("/api/knowledge/documents")

    assert response.status_code == 200
    documents = response.json()
    assert len(documents) == 2
    by_id = {item["document_id"]: item for item in documents}
    assert by_id["doc1"]["chunk_count"] == 2
    assert by_id["doc1"]["doc_name"] == "docs/one.md"
    assert by_id["doc1"]["embedding_model"] == "fake"
    assert by_id["doc2"]["chunk_count"] == 1


def test_knowledge_document_chunks_are_sorted() -> None:
    client = make_client()

    response = client.get("/api/knowledge/documents/doc1/chunks")

    assert response.status_code == 200
    chunks = response.json()
    assert [chunk["chunk_index"] for chunk in chunks] == [0, 1]
    assert chunks[0]["anchor_id"] == "doc1#0:aaaaaaaa"


def test_knowledge_chunk_by_anchor() -> None:
    client = make_client()

    response = client.get("/api/knowledge/chunks", params={"anchor_id": "doc1#1:bbbbbbbb"})

    assert response.status_code == 200
    body = response.json()
    assert body["text"] == "第二段"
    assert body["chunk_index"] == 1


def test_knowledge_missing_document_returns_404() -> None:
    client = make_client()

    response = client.get("/api/knowledge/documents/missing/chunks")

    assert response.status_code == 404


def test_knowledge_missing_anchor_returns_404() -> None:
    client = make_client()

    response = client.get("/api/knowledge/chunks", params={"anchor_id": "missing"})

    assert response.status_code == 404
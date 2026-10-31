import pytest

from app.vectorstore.base import ChunkRecord
from app.vectorstore.memory import InMemoryVectorStore


def test_search_returns_most_similar_record() -> None:
    store = InMemoryVectorStore()
    store.upsert([
        ChunkRecord(id="a", text="苹果", embedding=[1.0, 0.0, 0.0]),
        ChunkRecord(id="b", text="香蕉", embedding=[0.0, 1.0, 0.0]),
    ])

    results = store.search([1.0, 0.0, 0.0], top_k=1)

    assert results[0].id == "a"
    assert results[0].score == pytest.approx(1.0)


def test_search_returns_empty_for_empty_store() -> None:
    store = InMemoryVectorStore()
    assert store.search([1.0, 0.0], top_k=1) == []
def test_delete_by_document_id_removes_matching_chunks() -> None:
    store = InMemoryVectorStore()
    store.upsert(
        [
            ChunkRecord(
                id="a",
                text="苹果",
                embedding=[1.0, 0.0],
                metadata={"document_id": "doc-1"},
            ),
            ChunkRecord(
                id="b",
                text="香蕉",
                embedding=[0.0, 1.0],
                metadata={"document_id": "doc-2"},
            ),
        ]
    )

    store.delete_by_document_id("doc-1")

    assert [record.id for record in store.all_chunks()] == ["b"]
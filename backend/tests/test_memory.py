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

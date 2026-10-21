"""MilvusStore 集成测试。

需要本地 Milvus 正常运行。若未启动，测试会被跳过。
"""

import uuid

import pytest

from app.vectorstore.base import ChunkRecord
from app.vectorstore.milvus import MilvusStore


@pytest.mark.integration
def test_milvus_store_upsert_and_search() -> None:
    collection_name = f"test_chunks_{uuid.uuid4().hex[:8]}"
    store = MilvusStore(collection_name=collection_name)

    try:
        store.upsert(
            [
                ChunkRecord(
                    id="a",
                    text="苹果",
                    embedding=[1.0, 0.0, 0.0],
                    metadata={"source": "test"},
                ),
                ChunkRecord(
                    id="b",
                    text="香蕉",
                    embedding=[0.0, 1.0, 0.0],
                    metadata={"source": "test"},
                ),
            ]
        )

        results = store.search([1.0, 0.0, 0.0], top_k=1)

        assert len(results) == 1
        assert results[0].id == "a"
        assert results[0].text == "苹果"
        assert results[0].metadata["source"] == "test"
        assert results[0].score == pytest.approx(1.0)
    finally:
        store.drop_collection()
        store.close()
"""Milvus 向量库实现。"""

import json

from pymilvus import MilvusClient

from app.config import get_settings
from app.vectorstore.base import ChunkRecord, SearchResult, VectorStore


class MilvusStore(VectorStore):
    """基于 Milvus 的向量存储实现。"""

    def __init__(
        self,
        uri: str | None = None,
        collection_name: str | None = None,
        timeout: float | None = None,
    ) -> None:
        settings = get_settings()
        self.uri = uri or settings.milvus_uri
        self.collection_name = collection_name or settings.milvus_collection_name
        self.timeout = timeout if timeout is not None else settings.milvus_timeout
        self._client = MilvusClient(uri=self.uri, timeout=self.timeout)

    def upsert(self, records: list[ChunkRecord]) -> None:
        if not records:
            return

        dimension = self._validate_and_get_dimension(records)
        self._ensure_collection(dimension)

        data = [
            {
                "id": record.id,
                "vector": record.embedding,
                "text": record.text,
                "metadata_json": json.dumps(record.metadata, ensure_ascii=False),
            }
            for record in records
        ]
        self._client.upsert(
            collection_name=self.collection_name,
            data=data,
            timeout=self.timeout,
        )
        # 入库后刷新，保证紧随其后的检索能立即看到数据。
        self._client.flush(self.collection_name)

    def search(
        self,
        vector: list[float],
        top_k: int = 5,
        filter_expr: str | None = None,
    ) -> list[SearchResult]:
        if top_k <= 0 or not self._client.has_collection(self.collection_name):
            return []

        results = self._client.search(
            collection_name=self.collection_name,
            data=[vector],
            limit=top_k,
            filter=filter_expr or "",
            output_fields=["text", "metadata_json"],
            consistency_level="Strong",
            timeout=self.timeout,
        )

        if not results:
            return []

        search_results: list[SearchResult] = []
        for hit in results[0]:
            entity = hit.get("entity") or {}
            metadata_json = entity.get("metadata_json") or "{}"
            search_results.append(
                SearchResult(
                    id=str(hit["id"]),
                    score=float(hit["distance"]),
                    text=str(entity.get("text") or ""),
                    metadata=json.loads(metadata_json),
                )
            )
        return search_results

    def close(self) -> None:
        self._client.close()

    def drop_collection(self) -> None:
        """删除当前 Collection，主要用于测试和重建索引。"""
        if self._client.has_collection(self.collection_name):
            self._client.drop_collection(self.collection_name)

    def _ensure_collection(self, dimension: int) -> None:
        if self._client.has_collection(self.collection_name):
            return

        self._client.create_collection(
            collection_name=self.collection_name,
            dimension=dimension,
            primary_field_name="id",
            id_type="string",
            vector_field_name="vector",
            metric_type="COSINE",
            auto_id=False,
            enable_dynamic_field=True,
            max_length=512,
            timeout=self.timeout,
        )

    @staticmethod
    def _validate_and_get_dimension(records: list[ChunkRecord]) -> int:
        dimension = len(records[0].embedding)
        if dimension == 0:
            raise ValueError("embedding 不能为空")
        for record in records:
            if len(record.embedding) != dimension:
                raise ValueError("同一批 chunk 的 embedding 维度必须一致")
        return dimension
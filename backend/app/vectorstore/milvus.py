"""Milvus 向量库实现。"""

import json

from pymilvus import MilvusClient

from app.config import get_settings
from app.vectorstore.base import ChunkRecord, SearchResult, VectorStore,TextRecord


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
            consistency_level="Strong", #控制本次读取操作能看到多“新”的数据，Strong 等到最新数据同步后再查询，延迟最高。默认的 Bounded 级别允许读取略微滞后的数据
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

    def all_chunks(self, include_text: bool = True) -> list[TextRecord]:
        if not self._client.has_collection(self.collection_name):
            return []
        # include_text=False 供只遍历 metadata 的调用方减负：
        # 实测 2620 块下 text 约占载荷一半，不传输可把全表扫描
        # 从 ~618ms 降到 ~327ms（知识库文档列表场景）。
        output_fields = (
            ["id", "text", "metadata_json"]
            if include_text
            else ["id", "metadata_json"]
        )
        rows = self._client.query(
            collection_name=self.collection_name,
            filter="",
            output_fields=output_fields,
            limit=16384,
            # 刻意用默认（Bounded）一致性，理由同 get_by_ids：
            # upsert/delete 都已同步 flush，新数据对 Bounded 读立即可见，
            # Strong 每次要多等约 250ms 时间戳对齐（实测 874ms vs 618ms）。
            consistency_level="Bounded",
        )

        records: list[TextRecord] = []
        for row in rows:
            records.append(self._row_to_text_record(row))
        return records

    def get_by_ids(self, ids: list[str]) -> list[TextRecord]:
        """按主键直查，避免为了单条记录扫全表。

        这里刻意用默认（Bounded）一致性：``upsert`` 已同步 flush，
        新数据对 Bounded 读立即可见；而 Strong 每次要多等约 200ms 的
        时间戳对齐，对「读一条」这种请求不划算（实测 204ms vs 9ms）。
        """
        if not ids:
            return []
        if not self._client.has_collection(self.collection_name):
            return []

        rows = self._client.get(
            collection_name=self.collection_name,
            ids=ids,
            output_fields=["id", "text", "metadata_json"],
            timeout=self.timeout,
        )

        found = {str(row["id"]): self._row_to_text_record(row) for row in rows}
        # 主键查询不保证返回顺序，这里按入参顺序还原
        return [found[record_id] for record_id in ids if record_id in found]

    @staticmethod
    def _row_to_text_record(row: dict) -> TextRecord:
        return TextRecord(
            id=str(row["id"]),
            text=str(row.get("text") or ""),
            metadata=json.loads(row.get("metadata_json") or "{}"),
        )



    def delete_by_ids(self, ids: list[str]) -> None:
        if not ids:
            return
        if not self._client.has_collection(self.collection_name):
            return
        self._client.delete(
            collection_name=self.collection_name,
            ids=ids,
            timeout=self.timeout,
        )
        self._client.flush(self.collection_name)

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

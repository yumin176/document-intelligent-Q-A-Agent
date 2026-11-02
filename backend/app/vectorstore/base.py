"""向量库抽象接口与数据模型。"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class ChunkRecord:
    """一条待入库的 chunk 记录。"""

    id: str
    text: str
    embedding: list[float]
    metadata: dict = field(default_factory=dict)


@dataclass
class SearchResult:
    """一条检索命中结果。"""

    id: str
    score: float
    text: str
    metadata: dict = field(default_factory=dict)

@dataclass
class TextRecord:
    id: str
    text: str
    metadata: dict = field(default_factory=dict)

class VectorStore(ABC):
    """向量存储接口。

    上层检索模块只依赖这个接口，后续可以切换
    InMemory / Milvus / Chroma 等实现。
    """

    @abstractmethod
    def upsert(self, records: list[ChunkRecord]) -> None:
        """写入或更新若干条 chunk 记录。"""

    @abstractmethod
    def search(
        self,
        vector: list[float],
        top_k: int = 5,
        filter_expr: str | None = None,
    ) -> list[SearchResult]:
        """按向量相似度检索 top_k 条记录。"""

    @abstractmethod
    def all_chunks(self, include_text: bool = True) -> list[TextRecord]:
        """返回全部 chunk 的文本和元数据，用于构建 BM25 索引。

        include_text=False 时允许实现省略正文（如 Milvus 不传输 text
        字段），供只需要统计/遍历 metadata 的调用方（知识库文档列表、
        入库后清理旧锚点）减负。text 内容不保证有值。
        """

    def get_by_ids(self, ids: list[str]) -> list[TextRecord]:
        """按主键批量取回 chunk，返回顺序与入参一致，查不到的 id 直接跳过。

        默认实现退化为全量扫描；Milvus / 内存实现会覆盖成主键直查。
        """
        if not ids:
            return []
        wanted = set(ids)
        found = {record.id: record for record in self.all_chunks() if record.id in wanted}
        return [found[record_id] for record_id in ids if record_id in found]

    @abstractmethod
    def delete_by_ids(self, ids: list[str]) -> None:
        """按主键删除若干条 chunk。"""

    def delete_by_document_id(self, document_id: str) -> None:
        """删除属于指定文档的全部 chunk。"""
        ids = [
            record.id
            for record in self.all_chunks()
            if record.metadata.get("document_id") == document_id
        ]
        if ids:
            self.delete_by_ids(ids)

    def close(self) -> None:
        """释放底层连接，子类按需实现。"""

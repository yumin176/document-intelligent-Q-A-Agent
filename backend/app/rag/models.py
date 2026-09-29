"""RAG 服务的数据模型。"""

from dataclasses import dataclass


@dataclass
class Source:
    """一条引用来源。"""

    chunk_id: str
    score: float
    text: str
    metadata: dict


@dataclass
class RAGAnswer:
    """RAG 回答结果。"""

    answer: str
    sources: list[Source]
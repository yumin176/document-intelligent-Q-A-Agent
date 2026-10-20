from abc import ABC, abstractmethod

from app.vectorstore.base import SearchResult


class Retriever(ABC):
    @abstractmethod
    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        """根据自然语言查询返回 top_k 结果。"""

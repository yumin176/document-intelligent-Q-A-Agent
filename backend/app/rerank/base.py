from abc import ABC, abstractmethod

from app.vectorstore.base import SearchResult

class Reranker(ABC):
    @abstractmethod
    def rerank(
        self,
        query: str,
        candidates: list[SearchResult],
        top_k: int = 5,
    ) -> list[SearchResult]:
        """根据候选chunk返回精排后的前top_k条chunk"""

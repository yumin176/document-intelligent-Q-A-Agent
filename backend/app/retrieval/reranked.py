"""
HybridRetriever
→ 召回 candidate_k 条
→ 精排（DashScopeReranker）
→ 返回最终 top_k
"""

from app.rerank.base import Reranker
from app.retrieval.base import Retriever
from app.vectorstore.base import SearchResult


class RerankedRetriever(Retriever):
    def __init__(
        self,
        base_retriever: Retriever,
        reranker: Reranker,
        candidate_k: int = 10,
    ) -> None:
        if candidate_k<=0:
            raise ValueError("candidate_k不能小于等于0")
        self.base_retriever=base_retriever
        self.reranker=reranker
        self.candidate_k=candidate_k

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        """
        candidate_k <= 0 抛错；
        top_k <= 0 返回空，不调用 reranker；
        确保传给 Reranker 的候选数不少于最终 top_k。
        """
        if top_k<=0:
            return []
        candidate_k = max(self.candidate_k, top_k)
        candidates = self.base_retriever.search(query, top_k=candidate_k)
        return self.reranker.rerank(query, candidates, top_k=top_k)

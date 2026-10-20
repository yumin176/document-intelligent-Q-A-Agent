from app.rerank.base import Reranker
from app.vectorstore.base import SearchResult
# 基线，不做任何重排
class NoOpReranker(Reranker):
    def rerank(self, query: str, candidates: list[SearchResult], top_k: int = 5) -> list[SearchResult]:
        # top_k小于等于0返回空
        if top_k<=0:
            return []
        return candidates[:top_k]

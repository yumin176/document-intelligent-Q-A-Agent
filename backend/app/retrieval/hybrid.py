from app.retrieval.base import Retriever
from app.vectorstore.base import SearchResult

# 混合检索
class HybridRetriever(Retriever):
    def __init__(
        self,
        vector_retriever: Retriever,
        bm25_retriever: Retriever,
        rrf_k: int = 60,
        candidate_k: int = 10,
    ) -> None:
        if rrf_k <= 0:
            raise ValueError("rrf_k 必须大于 0")
        if candidate_k <= 0:
            raise ValueError("candidate_k 必须大于 0")
        self.vector_retriever=vector_retriever
        self.bm25_retriever=bm25_retriever
        self.rrf_k=rrf_k
        self.candidate_k=candidate_k

    def search(self, query: str,top_k: int=5) -> list[SearchResult]:
        # top_k不能小于等于0
        if top_k<=0:
            return []
        # 候选数量不少于最终返回数量
        candidate_k = max(self.candidate_k, top_k)
        kw_results=self.bm25_retriever.search(query,candidate_k)
        vec_results=self.vector_retriever.search(query,candidate_k)
        # 如果两个都是空数组，就返回空数组
        if (not kw_results) and (not vec_results):
            return []
        scores: dict[str, float] = {}
        results_by_id: dict[str, SearchResult] = {}
        for single_results in [kw_results,vec_results]:
            for index,result in enumerate(single_results,start=1):
                scores[result.id]=scores.get(result.id,0.0)+ 1.0 / (self.rrf_k + index)
                results_by_id[result.id] = result

        result=[
            SearchResult(
                id=chunk_id,
                score=score,
                text=results_by_id[chunk_id].text,
                metadata=results_by_id[chunk_id].metadata
            )
            for chunk_id,score in sorted(scores.items(),key=lambda item:item[1],reverse=True)
        ]

        return result[:top_k]

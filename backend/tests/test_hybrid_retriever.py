
from app.retrieval.base import Retriever
from app.retrieval.hybrid import HybridRetriever
from app.vectorstore.base import SearchResult
# 固定返回顺序
class FakeVectorRetriever(Retriever):
    def __init__(self):
        pass
    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        return [
            SearchResult(id="a",score=0.85,text="var声明的变量有变量提升"),
            SearchResult(id="b",score=0.75,text="var声明的变量可以重复声明"),
            SearchResult(id="c",score=0.51,text="const和let声明的变量不会变量提升")
        ]
class FakeBM25Retriever(Retriever):
    def __init__(self):
        pass
    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        return  [
            SearchResult(id="c",score=0.51,text="const和let声明的变量不会变量提升"),
            SearchResult(id="a",score=0.85,text="var声明的变量有变量提升"),
            SearchResult(id="b",score=0.75,text="var声明的变量可以重复声明"),
        ]

# 返回空数组
class EmptyVectorRetriever(Retriever):
    def __init__(self):
        pass
    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        return []

class EmptyBM25Retriever(Retriever):
    def __init__(self):
        pass
    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        return []

def test_hybrid_rrf_order() -> None:
    retriever = HybridRetriever(
        vector_retriever=FakeVectorRetriever(),
        bm25_retriever=FakeBM25Retriever(),
    )

    results = retriever.search("var", top_k=3)
    print(results)
    assert [result.id for result in results] == ["a", "c", "b"]
    assert all(result.score > 0.02 for result in results)

def test_hybrid_top_k_limit() -> None:
    retriever = HybridRetriever(
        vector_retriever=FakeVectorRetriever(),
        bm25_retriever=FakeBM25Retriever(),
    )
    assert len(retriever.search("var", top_k=2)) == 2


def test_hybrid_one_side_empty() -> None:
    # vector 有结果，bm25 返回空
    # 仍然应走 RRF，并遵守 top_k
    retriever = HybridRetriever(
        vector_retriever=FakeVectorRetriever(),
        bm25_retriever=EmptyBM25Retriever(),
    )
    assert [result.id for result in retriever.search("var")] == ["a", "b", "c"]


def test_hybrid_both_empty() -> None:
    # 两个都返回空 → []
    retriever = HybridRetriever(
        vector_retriever=EmptyVectorRetriever(),
        bm25_retriever=EmptyBM25Retriever(),
    )
    assert retriever.search("var") == []

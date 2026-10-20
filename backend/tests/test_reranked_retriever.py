import pytest

from app.rerank.base import Reranker
from app.retrieval.base import Retriever
from app.retrieval.reranked import RerankedRetriever
from app.vectorstore.base import SearchResult


def make_candidates() -> list[SearchResult]:
    return [
        SearchResult(
            id="a",
            score=0.83,
            text="var 声明的变量有变量提升",
            metadata={"filename": "js.md", "chunk_index": 0},
        ),
        SearchResult(
            id="b",
            score=0.73,
            text="var 声明的变量可以重复声明",
            metadata={"filename": "js.md", "chunk_index": 1},
        ),
        SearchResult(
            id="c",
            score=0.51,
            text="const 和 let 声明的变量不会变量提升",
            metadata={"filename": "js.md", "chunk_index": 2},
        ),
        SearchResult(
            id="d",
            score=0.61,
            text="在全局作用域下 var 声明的变量会被绑定到全局对象",
            metadata={"filename": "js.md", "chunk_index": 3},
        ),
        SearchResult(
            id="e",
            score=0.62,
            text="var 变量只用函数作用域和全局作用域",
            metadata={"filename": "js.md", "chunk_index": 4},
        ),
    ]


class SpyBaseRetriever(Retriever):
    def __init__(self) -> None:
        self.calls: list[int] = []

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        self.calls.append(top_k)
        return make_candidates()[:top_k]


class SpyReranker(Reranker):
    def __init__(self) -> None:
        self.calls: list[tuple[list[SearchResult], int]] = []

    def rerank(
        self,
        query: str,
        candidates: list[SearchResult],
        top_k: int = 5,
    ) -> list[SearchResult]:
        self.calls.append((candidates, top_k))
        return list(reversed(candidates))[:top_k]


def test_reranked_retriever_reranks_and_limits() -> None:
    base_retriever = SpyBaseRetriever()
    reranker = SpyReranker()
    retriever = RerankedRetriever(base_retriever, reranker, candidate_k=4)

    results = retriever.search("var 的作用域", top_k=3)

    assert [result.id for result in results] == ["d", "c", "b"]
    assert base_retriever.calls == [4]
    assert len(reranker.calls) == 1
    assert len(reranker.calls[0][0]) == 4
    assert reranker.calls[0][1] == 3


def test_candidate_k_expands_to_top_k() -> None:
    base_retriever = SpyBaseRetriever()
    reranker = SpyReranker()
    retriever = RerankedRetriever(base_retriever, reranker, candidate_k=2)

    retriever.search("var 的作用域", top_k=4)

    assert base_retriever.calls == [4]
    assert len(reranker.calls[0][0]) == 4


def test_non_positive_top_k_skips_dependencies() -> None:
    base_retriever = SpyBaseRetriever()
    reranker = SpyReranker()
    retriever = RerankedRetriever(base_retriever, reranker)

    assert retriever.search("var 的作用域", top_k=0) == []
    assert retriever.search("var 的作用域", top_k=-1) == []
    assert base_retriever.calls == []
    assert reranker.calls == []


def test_invalid_candidate_k_raises() -> None:
    with pytest.raises(ValueError):
        RerankedRetriever(SpyBaseRetriever(), SpyReranker(), candidate_k=0)

    with pytest.raises(ValueError):
        RerankedRetriever(SpyBaseRetriever(), SpyReranker(), candidate_k=-1)


def test_empty_candidates_return_empty() -> None:
    class EmptyRetriever(Retriever):
        def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
            return []

    base_retriever = EmptyRetriever()
    reranker = SpyReranker()
    retriever = RerankedRetriever(base_retriever, reranker)

    assert retriever.search("任意问题", top_k=3) == []
    assert len(reranker.calls) == 1
    assert reranker.calls[0][0] == []
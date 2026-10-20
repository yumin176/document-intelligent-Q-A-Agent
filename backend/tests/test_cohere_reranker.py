import json

import httpx
import pytest

from app.rerank.cohere import CohereReranker
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
    ]


def make_reranker(handler, **kwargs) -> CohereReranker:
    client = httpx.Client(transport=httpx.MockTransport(handler))
    return CohereReranker(
        api_url="https://example.com/rerank",
        api_key="test-key",
        model="cohere-rerank-v4.0-fast",
        client=client,
        **kwargs,
    )


def make_handler(results: list[dict], status_code: int = 200):
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["Authorization"] == "Bearer test-key"
        payload = json.loads(request.content)
        assert payload["model"] == "cohere-rerank-v4.0-fast"
        assert payload["query"] == "var 的作用域"
        assert payload["top_n"] == len(results)
        return httpx.Response(status_code, json={"results": results})

    return handler


def test_cohere_rerank_orders_and_preserves_metadata() -> None:
    reranker = make_reranker(
        make_handler(
            [
                {"index": 1, "relevance_score": 0.98},
                {"index": 0, "relevance_score": 0.72},
                {"index": 2, "relevance_score": 0.40},
            ]
        )
    )

    results = reranker.rerank("var 的作用域", make_candidates(), top_k=3)

    assert [item.id for item in results] == ["b", "a", "c"]
    assert results[0].metadata == {"filename": "js.md", "chunk_index": 1}


def test_cohere_rerank_respects_top_k() -> None:
    reranker = make_reranker(
        make_handler(
            [
                {"index": 1, "relevance_score": 0.98},
                {"index": 0, "relevance_score": 0.72},
            ]
        )
    )

    results = reranker.rerank("var 的作用域", make_candidates(), top_k=2)

    assert [item.id for item in results] == ["b", "a"]


def test_cohere_rerank_falls_back_on_http_error() -> None:
    reranker = make_reranker(make_handler([], status_code=500))

    results = reranker.rerank("var 的作用域", make_candidates(), top_k=2)

    assert [item.id for item in results] == ["a", "b"]


def test_cohere_rerank_falls_back_on_invalid_json() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="not json")

    reranker = make_reranker(handler)

    results = reranker.rerank("var 的作用域", make_candidates(), top_k=2)

    assert [item.id for item in results] == ["a", "b"]


def test_cohere_rerank_falls_back_on_invalid_index() -> None:
    reranker = make_reranker(
        make_handler(
            [
                {"index": 99, "relevance_score": 0.98},
                {"index": 0, "relevance_score": 0.72},
            ]
        )
    )

    results = reranker.rerank("var 的作用域", make_candidates(), top_k=2)

    assert [item.id for item in results] == ["a", "b"]


def test_cohere_rerank_falls_back_on_duplicate_index() -> None:
    reranker = make_reranker(
        make_handler(
            [
                {"index": 1, "relevance_score": 0.98},
                {"index": 1, "relevance_score": 0.72},
            ]
        )
    )

    results = reranker.rerank("var 的作用域", make_candidates(), top_k=2)

    assert [item.id for item in results] == ["a", "b"]


def test_cohere_rerank_falls_back_on_non_numeric_score() -> None:
    reranker = make_reranker(
        make_handler(
            [
                {"index": 1, "relevance_score": "high"},
                {"index": 0, "relevance_score": 0.72},
            ]
        )
    )

    results = reranker.rerank("var 的作用域", make_candidates(), top_k=2)

    assert [item.id for item in results] == ["a", "b"]


def test_cohere_rerank_empty_candidates() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        raise AssertionError("空候选不应该调用 API")

    reranker = make_reranker(handler)

    assert reranker.rerank("var 的作用域", []) == []


def test_cohere_rerank_non_positive_top_k() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        raise AssertionError("top_k<=0 不应该调用 API")

    reranker = make_reranker(handler)

    assert reranker.rerank("var 的作用域", make_candidates(), top_k=0) == []
    assert reranker.rerank("var 的作用域", make_candidates(), top_k=-1) == []


def test_cohere_rerank_requires_api_key() -> None:
    with pytest.raises(ValueError):
        CohereReranker(
            api_url="https://example.com/rerank",
            api_key="",
            model="cohere-rerank-v4.0-fast",
        )
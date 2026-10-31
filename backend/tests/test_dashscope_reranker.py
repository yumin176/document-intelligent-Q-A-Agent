import json

import httpx
import pytest

from app.rerank.dashscope import DashScopeReranker
from app.vectorstore.base import SearchResult


def make_candidates() -> list[SearchResult]:
    return [
        SearchResult(
            id="a",
            score=0.83,
            text="【成都银行(601838)｜2025年年度报告】营业收入 23,602,625 千元",
            metadata={"doc_name": "成都银行_2025年年度报告.pdf", "chunk_index": 0},
        ),
        SearchResult(
            id="b",
            score=0.73,
            text="【成都银行(601838)｜2025年年度报告】不良贷款率 0.68%",
            metadata={"doc_name": "成都银行_2025年年度报告.pdf", "chunk_index": 1},
        ),
        SearchResult(
            id="c",
            score=0.51,
            text="【极米科技(688696)｜2025年年度报告】归属于上市公司股东的净利润",
            metadata={"doc_name": "极米科技_2025年年度报告.pdf", "chunk_index": 2},
        ),
    ]


def make_reranker(handler, **kwargs) -> DashScopeReranker:
    client = httpx.Client(transport=httpx.MockTransport(handler))
    return DashScopeReranker(
        api_url="https://example.com/rerank",
        api_key="test-key",
        model="qwen3.7-text-rerank",
        client=client,
        **kwargs,
    )


def make_handler(results: list[dict], status_code: int = 200):
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["Authorization"] == "Bearer test-key"
        payload = json.loads(request.content)
        assert payload["model"] == "qwen3.7-text-rerank"
        assert payload["input"]["query"] == "成都银行营业收入"
        assert payload["parameters"]["top_n"] == len(results)
        assert len(payload["input"]["documents"]) == len(make_candidates())
        return httpx.Response(status_code, json={"output": {"results": results}})

    return handler


def test_dashscope_rerank_orders_and_preserves_metadata() -> None:
    reranker = make_reranker(
        make_handler(
            [
                {"index": 1, "relevance_score": 0.98},
                {"index": 0, "relevance_score": 0.72},
                {"index": 2, "relevance_score": 0.40},
            ]
        )
    )

    results = reranker.rerank("成都银行营业收入", make_candidates(), top_k=3)

    assert [item.id for item in results] == ["b", "a", "c"]
    assert results[0].metadata["doc_name"] == "成都银行_2025年年度报告.pdf"
    assert reranker.fallback_count == 0


def test_dashscope_rerank_respects_top_k() -> None:
    reranker = make_reranker(
        make_handler(
            [
                {"index": 1, "relevance_score": 0.98},
                {"index": 0, "relevance_score": 0.72},
            ]
        )
    )

    results = reranker.rerank("成都银行营业收入", make_candidates(), top_k=2)

    assert [item.id for item in results] == ["b", "a"]


def test_dashscope_rerank_truncates_to_max_candidates() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        assert len(payload["input"]["documents"]) == 2
        assert payload["parameters"]["top_n"] == 2
        return httpx.Response(
            200,
            json={
                "output": {
                    "results": [
                        {"index": 1, "relevance_score": 0.9},
                        {"index": 0, "relevance_score": 0.6},
                    ]
                }
            },
        )

    reranker = make_reranker(handler, max_candidates=2)

    results = reranker.rerank("成都银行营业收入", make_candidates(), top_k=2)

    assert [item.id for item in results] == ["b", "a"]


def test_dashscope_rerank_falls_back_on_http_error() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(502, text="Bad Gateway")

    reranker = make_reranker(handler)

    results = reranker.rerank("成都银行营业收入", make_candidates(), top_k=2)

    assert [item.id for item in results] == ["a", "b"]
    assert reranker.fallback_count == 1
    assert "502" in (reranker.last_fallback_error or "")


def test_dashscope_rerank_falls_back_on_invalid_json() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="not json")

    reranker = make_reranker(handler)

    results = reranker.rerank("成都银行营业收入", make_candidates(), top_k=2)

    assert [item.id for item in results] == ["a", "b"]
    assert reranker.fallback_count == 1


def test_dashscope_rerank_falls_back_on_missing_output() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"results": []})

    reranker = make_reranker(handler)

    results = reranker.rerank("成都银行营业收入", make_candidates(), top_k=2)

    assert [item.id for item in results] == ["a", "b"]
    assert reranker.fallback_count == 1


def test_dashscope_rerank_falls_back_on_invalid_index() -> None:
    reranker = make_reranker(
        make_handler(
            [
                {"index": 99, "relevance_score": 0.98},
                {"index": 0, "relevance_score": 0.72},
            ]
        )
    )

    results = reranker.rerank("成都银行营业收入", make_candidates(), top_k=2)

    assert [item.id for item in results] == ["a", "b"]
    assert reranker.fallback_count == 1


def test_dashscope_rerank_falls_back_on_duplicate_index() -> None:
    reranker = make_reranker(
        make_handler(
            [
                {"index": 1, "relevance_score": 0.98},
                {"index": 1, "relevance_score": 0.72},
            ]
        )
    )

    results = reranker.rerank("成都银行营业收入", make_candidates(), top_k=2)

    assert [item.id for item in results] == ["a", "b"]
    assert reranker.fallback_count == 1


def test_dashscope_rerank_falls_back_on_non_numeric_score() -> None:
    reranker = make_reranker(
        make_handler(
            [
                {"index": 1, "relevance_score": "high"},
                {"index": 0, "relevance_score": 0.72},
            ]
        )
    )

    results = reranker.rerank("成都银行营业收入", make_candidates(), top_k=2)

    assert [item.id for item in results] == ["a", "b"]
    assert reranker.fallback_count == 1


def test_dashscope_rerank_empty_candidates() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        raise AssertionError("空候选不应该调用 API")

    reranker = make_reranker(handler)

    assert reranker.rerank("成都银行营业收入", []) == []
    assert reranker.fallback_count == 0


def test_dashscope_rerank_non_positive_top_k() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        raise AssertionError("top_k<=0 不应该调用 API")

    reranker = make_reranker(handler)

    assert reranker.rerank("成都银行营业收入", make_candidates(), top_k=0) == []
    assert reranker.rerank("成都银行营业收入", make_candidates(), top_k=-1) == []


def test_dashscope_rerank_requires_api_key() -> None:
    with pytest.raises(ValueError):
        DashScopeReranker(
            api_url="https://example.com/rerank",
            api_key="",
            model="qwen3.7-text-rerank",
        )

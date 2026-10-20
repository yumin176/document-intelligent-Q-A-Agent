from app.llm.fake import FakeChatModel
from app.rerank.llm import LLMReranker
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


def make_reranker(response: str, max_candidates: int = 10) -> LLMReranker:
    return LLMReranker(
        chat_model=FakeChatModel(response=response),
        max_candidates=max_candidates,
    )


def test_rerank_orders_by_llm_scores() -> None:
    reranker = make_reranker(
        '[{"id":"b","score":9.0},{"id":"a","score":3.0},{"id":"c","score":1.0}]'
    )

    results = reranker.rerank("var 的作用域是什么", make_candidates())

    assert [item.id for item in results] == ["b", "a", "c"]
    assert [item.score for item in results] == [9.0, 3.0, 1.0]


def test_rerank_respects_top_k() -> None:
    reranker = make_reranker(
        '[{"id":"b","score":9.0},{"id":"a","score":3.0},{"id":"c","score":1.0}]'
    )

    results = reranker.rerank("var 的作用域是什么", make_candidates(), top_k=2)

    assert [item.id for item in results] == ["b", "a"]


def test_rerank_accepts_markdown_json() -> None:
    reranker = make_reranker(
        '```json\n'
        '[{"id":"b","score":9.0},{"id":"a","score":3.0},{"id":"c","score":1.0}]'
        '\n```'
    )

    results = reranker.rerank("var 的作用域是什么", make_candidates())

    assert [item.id for item in results] == ["b", "a", "c"]


def test_rerank_falls_back_on_invalid_json() -> None:
    reranker = make_reranker("模型没有返回 JSON")

    results = reranker.rerank("var 的作用域是什么", make_candidates(), top_k=3)

    assert [item.id for item in results] == ["a", "b", "c"]


def test_rerank_falls_back_on_non_numeric_score() -> None:
    reranker = make_reranker(
        '[{"id":"b","score":"很高"},{"id":"a","score":3.0},{"id":"c","score":1.0}]'
    )

    results = reranker.rerank("var 的作用域是什么", make_candidates(), top_k=3)

    assert [item.id for item in results] == ["a", "b", "c"]


def test_rerank_falls_back_on_unknown_id() -> None:
    reranker = make_reranker(
        '[{"id":"b","score":9.0},{"id":"unknown","score":3.0},{"id":"c","score":1.0}]'
    )

    results = reranker.rerank("var 的作用域是什么", make_candidates(), top_k=3)

    assert [item.id for item in results] == ["a", "b", "c"]


def test_rerank_falls_back_on_duplicate_id() -> None:
    reranker = make_reranker(
        '[{"id":"b","score":9.0},{"id":"a","score":3.0},{"id":"a","score":1.0}]'
    )

    results = reranker.rerank("var 的作用域是什么", make_candidates(), top_k=3)

    assert [item.id for item in results] == ["a", "b", "c"]


def test_rerank_falls_back_on_incomplete_output() -> None:
    reranker = make_reranker(
        '[{"id":"b","score":9.0},{"id":"a","score":3.0}]'
    )

    results = reranker.rerank("var 的作用域是什么", make_candidates(), top_k=3)

    assert [item.id for item in results] == ["a", "b", "c"]


def test_rerank_preserves_metadata() -> None:
    reranker = make_reranker(
        '[{"id":"b","score":9.0},{"id":"a","score":3.0},{"id":"c","score":1.0}]'
    )

    results = reranker.rerank("var 的作用域是什么", make_candidates())

    assert results[0].metadata == {"filename": "js.md", "chunk_index": 1}


def test_rerank_empty_candidates() -> None:
    reranker = make_reranker('[{"id":"a","score":9.0}]')

    assert reranker.rerank("任意问题", []) == []


def test_rerank_non_positive_top_k() -> None:
    reranker = make_reranker('[{"id":"a","score":9.0}]')

    assert reranker.rerank("任意问题", make_candidates(), top_k=0) == []
    assert reranker.rerank("任意问题", make_candidates(), top_k=-1) == []
import pytest

from app.agent.graph import (
    AgentRetrievalGraph,
    heuristic_route,
    parse_relevant_indices,
    parse_rewrite_query,
    parse_route,
)
from app.agent.service import AgentService, NO_RELEVANT_ANSWER
from app.llm.fake import ScriptedChatModel
from app.retrieval.base import Retriever
from app.vectorstore.base import SearchResult


def make_hit(chunk_id: str, text: str, chunk_index: int = 0) -> SearchResult:
    return SearchResult(
        id=chunk_id,
        score=0.9,
        text=text,
        metadata={"filename": "js.md", "chunk_index": chunk_index},
    )


class StaticRetriever(Retriever):
    def __init__(self, results: list[SearchResult]) -> None:
        self.results = results
        self.calls: list[tuple[str, int]] = []

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        self.calls.append((query, top_k))
        return self.results[:top_k]


class QueryAwareRetriever(Retriever):
    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        self.calls.append((query, top_k))
        if len(self.calls) == 1:
            return [make_hit("wrong", "无关内容", 9)]
        return [make_hit("right", "相关内容", 0)]


class FailingRetriever(Retriever):
    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        raise AssertionError("direct 路由不应调用检索器")


def make_service(
    responses: list[str],
    retriever: Retriever,
    max_rewrites: int = 1,
) -> tuple[AgentService, ScriptedChatModel]:
    model = ScriptedChatModel(responses)
    graph = AgentRetrievalGraph(
        retriever=retriever,
        chat_model=model,
        max_rewrites=max_rewrites,
        top_k=3,
    )
    return AgentService(graph=graph, chat_model=model, top_k=3), model


def test_parse_agent_outputs() -> None:
    assert parse_route('```json\n{"route":"direct"}\n```') == "direct"
    assert parse_route('说明文字 {"route":"retrieve"}') == "retrieve"
    assert parse_relevant_indices('{"relevant_indices":[1,3,3]}', 3) == [1, 3]
    assert parse_relevant_indices('{"relevant_indices":[]}', 3) == []
    assert parse_rewrite_query('{"query":"新的查询"}') == "新的查询"
    assert parse_rewrite_query("```text\n改写查询\n```") == "改写查询"
    assert heuristic_route("你好") == "direct"
    assert heuristic_route("var 的变量提升是什么") == "retrieve"


def test_invalid_agent_outputs_raise() -> None:
    with pytest.raises(ValueError):
        parse_route("not json")
    with pytest.raises(ValueError):
        parse_relevant_indices('{"relevant_indices":[4]}', 3)
    with pytest.raises(ValueError):
        parse_rewrite_query("   ")

def test_direct_route_skips_retrieval() -> None:
    service, model = make_service(
        ['{"route":"direct"}', "你好，我是文档问答助手。"],
        FailingRetriever(),
    )

    result = service.answer("你好")

    assert result.answer == "你好，我是文档问答助手。"
    assert result.sources == []
    assert len(model.calls) == 2


def test_retrieve_grade_and_answer_uses_relevant_only() -> None:
    first = make_hit("a", "无关内容", 1)
    second = make_hit("b", "相关内容", 0)
    service, model = make_service(
        [
            '{"route":"retrieve"}',
            '{"relevant_indices":[2]}',
            "相关内容 [1]",
        ],
        StaticRetriever([first, second]),
    )

    result = service.answer("测试问题")

    assert result.answer == "相关内容 [1]"
    assert [source.chunk_id for source in result.sources] == ["b"]
    assert "a" not in model.calls[-1][0].content


def test_rewrite_then_retrieve_again() -> None:
    retriever = QueryAwareRetriever()
    service, model = make_service(
        [
            '{"route":"retrieve"}',
            '{"relevant_indices":[]}',
            '{"query":"改写后的查询"}',
            '{"relevant_indices":[1]}',
            "恢复成功 [1]",
        ],
        retriever,
    )

    result = service.answer("原始问题")

    assert result.answer == "恢复成功 [1]"
    assert [source.chunk_id for source in result.sources] == ["right"]
    assert [query for query, _ in retriever.calls] == ["原始问题", "改写后的查询"]
    assert len(model.calls) == 5


def test_give_up_after_max_rewrites() -> None:
    retriever = StaticRetriever([make_hit("wrong", "无关内容", 9)])
    service, model = make_service(
        [
            '{"route":"retrieve"}',
            '{"relevant_indices":[]}',
            '{"query":"改写后的查询"}',
            '{"relevant_indices":[]}',
        ],
        retriever,
    )

    result = service.answer("原始问题")

    assert result.answer == NO_RELEVANT_ANSWER
    assert result.sources == []
    assert len(retriever.calls) == 2
    assert len(model.calls) == 4


def test_invalid_grader_falls_back_to_all_candidates() -> None:
    hits = [make_hit("a", "内容 A", 0), make_hit("b", "内容 B", 1)]
    service, _ = make_service(
        [
            '{"route":"retrieve"}',
            "不是 JSON",
            "回答 [1][2]",
        ],
        StaticRetriever(hits),
    )

    result = service.answer("测试问题")

    assert [source.chunk_id for source in result.sources] == ["a", "b"]
    assert result.answer == "回答 [1][2]"


def test_agent_stream_returns_sources_and_tokens() -> None:
    service, _ = make_service(
        [
            '{"route":"retrieve"}',
            '{"relevant_indices":[1]}',
            "流式回答",
        ],
        StaticRetriever([make_hit("a", "流式内容", 0)]),
    )

    tokens, sources = service.answer_stream("测试问题")

    assert "".join(tokens) == "流式回答"
    assert [source.chunk_id for source in sources] == ["a"]
from app.agent.graph import AgentGraph
from app.agent.service import AgentService, NO_RELEVANT_ANSWER
from app.llm.fake import ScriptedChatModel
from app.retrieval.base import Retriever
from app.vectorstore.base import SearchResult


def make_hit(chunk_id: str, text: str) -> SearchResult:
    return SearchResult(
        id=chunk_id,
        score=0.9,
        text=text,
        metadata={"filename": "js.md", "chunk_index": 0},
    )


class FakeRetriever(Retriever):
    def __init__(self, results: list[SearchResult]) -> None:
        self.results = results
        self.calls: list[tuple[str, int]] = []

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        self.calls.append((query, top_k))
        return self.results[:top_k]


class FailingRetriever(Retriever):
    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        raise AssertionError("direct 路径不应调用检索器")


def make_service(
    model: ScriptedChatModel,
    retriever: Retriever,
    top_k: int = 2,
) -> AgentService:
    graph = AgentGraph(
        chat_model=model,
        retriever=retriever,
        top_k=top_k,
    )
    return AgentService(graph=graph, chat_model=model, top_k=top_k)


def test_direct_answer_has_no_sources() -> None:
    model = ScriptedChatModel(
        [
            '{"route":"direct"}',
            "你好，我是文档问答助手。",
        ]
    )
    service = make_service(model, FailingRetriever())

    result = service.answer("你好")

    assert result.answer == "你好，我是文档问答助手。"
    assert result.sources == []
    assert len(model.calls) == 2


def test_retrieve_answer_uses_filtered_sources_and_top_k_override() -> None:
    retriever = FakeRetriever(
        [
            make_hit("a", "无关内容"),
            make_hit("b", "相关内容"),
        ]
    )
    model = ScriptedChatModel(
        [
            '{"route":"retrieve"}',
            '{"relevant_indices":[1]}',
            "相关内容 [1]",
        ]
    )
    service = make_service(model, retriever)

    result = service.answer("测试问题", top_k=1)

    assert result.answer == "相关内容 [1]"
    assert [source.chunk_id for source in result.sources] == ["a"]
    assert retriever.calls == [("测试问题", 1)]
    assert "无关内容" in model.calls[-1][0].content


def test_no_relevant_chunks_returns_refusal_without_generation() -> None:
    retriever = FakeRetriever([make_hit("wrong", "无关内容")])
    model = ScriptedChatModel(
        [
            '{"route":"retrieve"}',
            '{"relevant_indices":[]}',
            '{"query":"改写后的查询"}',
            '{"relevant_indices":[]}',
        ]
    )
    service = make_service(model, retriever)

    result = service.answer("原始问题")

    assert result.answer == NO_RELEVANT_ANSWER
    assert result.sources == []
    assert len(model.calls) == 4


def test_answer_rejects_non_positive_top_k() -> None:
    model = ScriptedChatModel([])
    service = make_service(model, FailingRetriever())

    try:
        service.answer("测试", top_k=0)
    except ValueError as exc:
        assert "top_k" in str(exc)
    else:
        raise AssertionError("top_k=0 应抛出 ValueError")

def test_stream_direct_answer_has_no_sources() -> None:
    model = ScriptedChatModel(
        [
            '{"route":"direct"}',
            "你好",
        ]
    )
    service = make_service(model, FailingRetriever())

    tokens, sources = service.answer_stream("你好")

    assert "".join(tokens) == "你好"
    assert sources == []
    assert len(model.calls) == 2


def test_stream_retrieve_answer_returns_sources_and_uses_top_k() -> None:
    retriever = FakeRetriever(
        [
            make_hit("a", "相关内容"),
            make_hit("b", "额外内容"),
        ]
    )
    model = ScriptedChatModel(
        [
            '{"route":"retrieve"}',
            '{"relevant_indices":[1]}',
            "流式回答",
        ]
    )
    service = make_service(model, retriever)

    tokens, sources = service.answer_stream("测试问题", top_k=1)

    assert "".join(tokens) == "流式回答"
    assert [source.chunk_id for source in sources] == ["a"]
    assert retriever.calls == [("测试问题", 1)]


def test_stream_refusal_yields_single_complete_message() -> None:
    retriever = FakeRetriever([make_hit("wrong", "无关内容")])
    model = ScriptedChatModel(
        [
            '{"route":"retrieve"}',
            '{"relevant_indices":[]}',
            '{"query":"改写后的查询"}',
            '{"relevant_indices":[]}',
        ]
    )
    service = make_service(model, retriever)

    tokens, sources = service.answer_stream("原始问题")

    assert list(tokens) == [NO_RELEVANT_ANSWER]
    assert sources == []
    assert len(model.calls) == 4
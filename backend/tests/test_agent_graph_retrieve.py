from app.agent.graph import AgentGraph
from app.llm.fake import FakeChatModel
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
    def __init__(self, results):
        self.results = results
        self.calls = []

    def search(self, query: str, top_k: int = 5):
        self.calls.append((query, top_k))
        return self.results[:top_k]


class FailingRetriever(Retriever):
    def search(self, query: str, top_k: int = 5):
        raise AssertionError("direct 路径不应该调用检索器")


def test_direct_route_skips_retrieval():
    retriever = FailingRetriever()
    graph = AgentGraph(
        chat_model=FakeChatModel('{"route":"direct"}'),
        retriever=retriever,
        top_k=2,
    )

    state = graph.run("你好")

    assert state["route"] == "direct"
    assert state["candidate_chunks"] == []


def test_retrieve_route_returns_chunks_and_counts():
    retriever = FakeRetriever(
        [
            make_hit("a", "var 有变量提升"),
            make_hit("b", "var 可以重复声明"),
        ]
    )
    graph = AgentGraph(
        chat_model=FakeChatModel('{"route":"retrieve"}'),
        retriever=retriever,
        top_k=2,
    )

    state = graph.run("var 有什么特点")
    assert state["route"] == "retrieve"
    assert [chunk.id for chunk in state["candidate_chunks"]] == ["a", "b"]
    assert state["retrieval_attempts"] == 1
    assert state["query_history"] == ["var 有什么特点"]
    assert retriever.calls == [("var 有什么特点", 2)]


def test_retrieve_route_deduplicates_chunks():
    retriever = FakeRetriever(
        [
            make_hit("a", "内容 A"),
            make_hit("a", "内容 A 重复"),
            make_hit("b", "内容 B"),
        ]
    )
    graph = AgentGraph(
        chat_model=FakeChatModel('{"route":"retrieve"}'),
        retriever=retriever,
        top_k=3,
    )

    state = graph.run("测试")

    assert [chunk.id for chunk in state["candidate_chunks"]] == ["a", "b"]


def test_route_wrapper_supports_retrieve_path():
    retriever = FakeRetriever([make_hit("a", "相关内容")])
    graph = AgentGraph(
        chat_model=FakeChatModel('{"route":"retrieve"}'),
        retriever=retriever,
        top_k=2,
    )

    assert graph.route("测试问题") == "retrieve"

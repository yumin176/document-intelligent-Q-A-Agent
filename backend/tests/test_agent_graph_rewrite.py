from app.agent.graph import AgentGraph
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


class QueryAwareRetriever(Retriever):
    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        self.calls.append((query, top_k))
        if len(self.calls) == 1:
            return [make_hit("wrong", "无关内容")]
        return [make_hit("right", "相关内容")]


class AlwaysIrrelevantRetriever(Retriever):
    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        return [make_hit("wrong", "无关内容")]


def test_rewrite_recovers_after_second_retrieval() -> None:
    retriever = QueryAwareRetriever()
    model = ScriptedChatModel(
        [
            '{"route":"retrieve"}',
            '{"relevant_indices":[]}',
            '{"query":"改写后的查询"}',
            '{"relevant_indices":[1]}',
        ]
    )
    graph = AgentGraph(chat_model=model, retriever=retriever, top_k=2)

    state = graph.run("原始问题")

    assert state["retrieval_attempts"] == 2
    assert state["rewrite_count"] == 1
    assert state["search_query"] == "改写后的查询"
    assert state["query_history"] == ["原始问题", "改写后的查询"]
    assert [chunk.id for chunk in state["relevant_chunks"]] == ["right"]
    assert retriever.calls == [("原始问题", 2), ("改写后的查询", 2)]


def test_rewrite_stops_at_limit_when_still_irrelevant() -> None:
    retriever = AlwaysIrrelevantRetriever()
    model = ScriptedChatModel(
        [
            '{"route":"retrieve"}',
            '{"relevant_indices":[]}',
            '{"query":"改写后的查询"}',
            '{"relevant_indices":[]}',
        ]
    )
    graph = AgentGraph(chat_model=model, retriever=retriever, top_k=2)

    state = graph.run("原始问题")

    assert state["retrieval_attempts"] == 2
    assert state["rewrite_count"] == 1
    assert state["relevant_chunks"] == []
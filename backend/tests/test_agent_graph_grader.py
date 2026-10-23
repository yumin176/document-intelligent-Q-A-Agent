from app.agent.graph import AgentGraph
from app.llm.fake import ScriptedChatModel
from app.retrieval.base import Retriever
from app.vectorstore.base import SearchResult


def make_hit(chunk_id: str) -> SearchResult:
    return SearchResult(
        id=chunk_id,
        score=0.9,
        text=f"内容 {chunk_id}",
        metadata={"filename": "js.md", "chunk_index": 0},
    )


class FakeRetriever(Retriever):
    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        return [make_hit("a"), make_hit("b"), make_hit("c")][:top_k]


def test_graph_filters_candidates_with_grader() -> None:
    model = ScriptedChatModel(
        [
            '{"route":"retrieve"}',
            '{"relevant_indices":[3,1]}',
        ]
    )
    graph = AgentGraph(chat_model=model, retriever=FakeRetriever(), top_k=3)

    state = graph.run("测试问题")

    assert [chunk.id for chunk in state["candidate_chunks"]] == ["a", "b", "c"]
    assert [chunk.id for chunk in state["relevant_chunks"]] == ["a", "c"]


def test_graph_keeps_all_candidates_on_invalid_grader_output() -> None:
    model = ScriptedChatModel(
        [
            '{"route":"retrieve"}',
            "not json",
        ]
    )
    graph = AgentGraph(chat_model=model, retriever=FakeRetriever(), top_k=3)

    state = graph.run("测试问题")

    assert [chunk.id for chunk in state["relevant_chunks"]] == ["a", "b", "c"]
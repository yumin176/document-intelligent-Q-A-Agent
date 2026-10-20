from app.llm.fake import FakeChatModel
from app.rag.service import RAGService
from app.retrieval.base import Retriever
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


class FakeRetriever(Retriever):
    def __init__(self, results: list[SearchResult]) -> None:
        self.results = results
        self.calls: list[tuple[str, int]] = []

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        self.calls.append((query, top_k))
        return self.results[:top_k]


def test_rag_service_uses_retriever_and_returns_sources() -> None:
    retriever = FakeRetriever(make_candidates())
    chat_model = FakeChatModel(response="员工每年有 10 天年假 [1]")
    service = RAGService(retriever, chat_model, top_k=2)

    result = service.answer("公司年假有多少天")

    assert retriever.calls == [("公司年假有多少天", 2)]
    assert result.answer == "员工每年有 10 天年假 [1]"
    assert [source.chunk_id for source in result.sources] == ["a", "b"]

    system_message = chat_model.calls[0][0]
    assert system_message.role == "system"
    assert "var 声明的变量有变量提升" in system_message.content
    assert chat_model.calls[0][1].content == "公司年假有多少天"


def test_rag_service_stream_returns_sources_and_tokens() -> None:
    retriever = FakeRetriever(make_candidates())
    chat_model = FakeChatModel(response="违约金 1000 元 [1]")
    service = RAGService(retriever, chat_model, top_k=2)

    tokens, sources = service.answer_stream("违约金是多少")

    assert [source.chunk_id for source in sources] == ["a", "b"]
    assert "".join(tokens) == "违约金 1000 元 [1]"


def test_rag_service_empty_results() -> None:
    retriever = FakeRetriever([])
    chat_model = FakeChatModel(response="资料中未找到相关信息 [1]")
    service = RAGService(retriever, chat_model, top_k=2)

    result = service.answer("公司年假有多少天")

    assert result.sources == []
    assert result.answer == "资料中未找到相关信息 [1]"
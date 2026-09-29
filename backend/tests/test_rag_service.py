"""RAGService 编排逻辑的单元测试。"""

from app.embedding.fake import FakeEmbedder
from app.llm.fake import FakeChatModel
from app.rag.service import RAGService
from app.vectorstore.base import ChunkRecord
from app.vectorstore.memory import InMemoryVectorStore


def test_rag_service_returns_answer_and_sources() -> None:
    embedder = FakeEmbedder(dim=8)
    store = InMemoryVectorStore()
    store.upsert(
        [
            ChunkRecord(
                id="doc:0",
                text="员工每年享有 10 天带薪年假。",
                embedding=[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                metadata={"filename": "员工手册.md", "chunk_index": 0},
            )
        ]
    )
    chat_model = FakeChatModel(response="员工每年有 10 天年假 [1]")

    service = RAGService(embedder, store, chat_model, top_k=2)
    result = service.answer("公司年假有多少天")

    assert result.answer == "员工每年有 10 天年假 [1]"
    assert len(result.sources) == 1
    assert result.sources[0].chunk_id == "doc:0"
    assert result.sources[0].metadata["filename"] == "员工手册.md"


def test_rag_service_injects_context_into_prompt() -> None:
    embedder = FakeEmbedder(dim=8)
    store = InMemoryVectorStore()
    store.upsert(
        [
            ChunkRecord(
                id="doc:0",
                text="合同条款：违约金为 1000 元。",
                embedding=[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                metadata={"filename": "合同.md"},
            )
        ]
    )
    chat_model = FakeChatModel(response="违约金 1000 元 [1]")

    service = RAGService(embedder, store, chat_model, top_k=2)
    service.answer("违约金是多少")

    system_message = chat_model.calls[0][0]
    assert system_message.role == "system"
    assert "合同条款：违约金为 1000 元。" in system_message.content
    assert "[1] 来源: 合同.md" in system_message.content
    assert chat_model.calls[0][1].content == "违约金是多少"
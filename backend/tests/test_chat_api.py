from fastapi.testclient import TestClient

from app.agent.graph import AgentGraph
from app.agent.service import AgentService
from app.dependencies import get_agent_service
from app.embedding.fake import FakeEmbedder
from app.llm.fake import ScriptedChatModel
from app.main import create_app
from app.retrieval.vector import VectorRetriever
from app.vectorstore.base import ChunkRecord
from app.vectorstore.memory import InMemoryVectorStore


def make_test_service() -> AgentService:
    store = InMemoryVectorStore()
    store.upsert(
        [
            ChunkRecord(
                id="doc:0",
                text="员工每年有 10 天带薪年假",
                embedding=[1.0, 0.0, 0.0],
                metadata={"filename": "员工手册.md", "chunk_index": 0},
            )
        ]
    )
    model = ScriptedChatModel(
        [
            '{"route":"retrieve"}',
            '{"relevant_indices":[1]}',
            "fake answer",
        ]
    )
    graph = AgentGraph(
        chat_model=model,
        retriever=VectorRetriever(
            embedder=FakeEmbedder(dim=3),
            vector_store=store,
        ),
        top_k=2,
    )
    return AgentService(graph=graph, chat_model=model, top_k=2)


def test_chat_endpoint() -> None:
    app = create_app()
    app.dependency_overrides[get_agent_service] = make_test_service
    client = TestClient(app)

    response = client.post("/api/chat", json={"question": "测试"})

    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == "fake answer"
    source = data["sources"][0]
    assert source["chunk_id"] == "doc:0"
    # 定位字段缺失时回退：anchor_id 取 chunk_id，filename / chunk_index 从 metadata 带出
    assert source["anchor_id"] == "doc:0"
    assert source["filename"] == "员工手册.md"
    assert source["chunk_index"] == 0
    assert source["document_id"] == ""


def test_chat_stream_endpoint() -> None:
    app = create_app()
    app.dependency_overrides[get_agent_service] = make_test_service
    client = TestClient(app)

    with client.stream("POST", "/api/chat/stream", json={"question": "测试"}) as response:
        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]
        body = "".join(response.iter_text())

    assert "event: sources" in body
    assert '"chunk_id": "doc:0"' in body
    assert '"anchor_id": "doc:0"' in body
    assert "event: delta" in body
    assert "event: done" in body
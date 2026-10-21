"""依赖注入。"""

from collections.abc import Callable
from functools import lru_cache

from app.agent.graph import AgentRetrievalGraph
from app.agent.service import AgentService
from app.config import get_settings
from app.embedding.base import Embedder
from app.embedding.openai import OpenAIEmbedder
from app.ingestion.service import IngestionService
from app.llm.base import ChatModel
from app.llm.openai import OpenAIChatModel
from app.rag.service import RAGService
from app.rerank.cohere import CohereReranker
from app.retrieval.base import Retriever
from app.retrieval.bm25 import BM25Retriever
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.reranked import RerankedRetriever
from app.retrieval.vector import VectorRetriever
from app.vectorstore.base import VectorStore
from app.vectorstore.milvus import MilvusStore


@lru_cache
def get_embedder() -> Embedder:
    return OpenAIEmbedder()


@lru_cache
def get_vector_store() -> VectorStore:
    return MilvusStore()


@lru_cache
def get_chat_model() -> ChatModel:
    return OpenAIChatModel()


@lru_cache
def get_bm25_retriever() -> BM25Retriever:
    return BM25Retriever(get_vector_store())


def get_bm25_refresher() -> Callable[[], None]:
    """按当前检索模式返回 BM25 刷新函数，便于接口测试注入替身。"""
    if get_settings().retrieval_mode not in {"bm25", "hybrid", "hybrid_rerank"}:
        return lambda: None
    return get_bm25_retriever().refresh


@lru_cache
def get_retriever() -> Retriever:
    settings = get_settings()
    mode = settings.retrieval_mode
    candidate_k = settings.retrieval_candidate_k

    if candidate_k <= 0:
        raise ValueError("retrieval_candidate_k 必须大于 0")

    vector_store = get_vector_store()
    vector_retriever = VectorRetriever(get_embedder(), vector_store)

    if mode == "vector":
        return vector_retriever
    if mode == "bm25":
        return get_bm25_retriever()
    if mode == "hybrid":
        return HybridRetriever(
            vector_retriever=vector_retriever,
            bm25_retriever=get_bm25_retriever(),
            candidate_k=candidate_k,
        )
    if mode == "hybrid_rerank":
        hybrid_retriever = HybridRetriever(
            vector_retriever=vector_retriever,
            bm25_retriever=get_bm25_retriever(),
            candidate_k=candidate_k,
        )
        return RerankedRetriever(
            base_retriever=hybrid_retriever,
            reranker=CohereReranker(),
            candidate_k=candidate_k,
        )
    raise ValueError(f"不支持的 retrieval_mode: {mode}")


@lru_cache
def get_rag_service() -> RAGService:
    return RAGService(
        retriever=get_retriever(),
        chat_model=get_chat_model(),
    )


@lru_cache
def get_agent_service() -> AgentService:
    settings = get_settings()
    if settings.agent_max_rewrites < 0:
        raise ValueError("agent_max_rewrites 不能小于 0")

    chat_model = get_chat_model()
    graph = AgentRetrievalGraph(
        retriever=get_retriever(),
        chat_model=chat_model,
        max_rewrites=settings.agent_max_rewrites,
    )
    return AgentService(
        graph=graph,
        chat_model=chat_model,
    )


@lru_cache
def get_ingestion_service() -> IngestionService:
    return IngestionService(
        embedder=get_embedder(),
        vector_store=get_vector_store(),
    )
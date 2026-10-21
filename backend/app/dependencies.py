"""依赖注入。"""

from functools import lru_cache

from app.config import get_settings
from app.embedding.base import Embedder
from app.embedding.openai import OpenAIEmbedder
from app.ingestion.service import IngestionService
from app.llm.base import ChatModel
from app.llm.openai import OpenAIChatModel
from app.rag.service import RAGService
from app.rerank.cohere import CohereReranker
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


@lru_cache
def get_rag_service() -> RAGService:
    settings = get_settings()
    mode = settings.retrieval_mode
    candidate_k = settings.retrieval_candidate_k

    if candidate_k <= 0:
        raise ValueError("retrieval_candidate_k 必须大于 0")

    vector_store = get_vector_store()
    vector_retriever = VectorRetriever(get_embedder(), vector_store)

    if mode == "vector":
        retriever = vector_retriever
    elif mode == "bm25":
        retriever = get_bm25_retriever()
    elif mode == "hybrid":
        retriever = HybridRetriever(
            vector_retriever=vector_retriever,
            bm25_retriever=get_bm25_retriever(),
            candidate_k=candidate_k,
        )
    elif mode == "hybrid_rerank":
        hybrid_retriever = HybridRetriever(
            vector_retriever=vector_retriever,
            bm25_retriever=get_bm25_retriever(),
            candidate_k=candidate_k,
        )
        retriever = RerankedRetriever(
            base_retriever=hybrid_retriever,
            reranker=CohereReranker(),
            candidate_k=candidate_k,
        )
    else:
        raise ValueError(f"不支持的 retrieval_mode: {mode}")

    return RAGService(
        retriever=retriever,
        chat_model=get_chat_model(),
    )


@lru_cache
def get_ingestion_service() -> IngestionService:
    return IngestionService(
        embedder=get_embedder(),
        vector_store=get_vector_store(),
    )
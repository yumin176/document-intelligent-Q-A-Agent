"""
依赖注入
"""
from functools import lru_cache
from app.ingestion.service import IngestionService
from app.rag.service import RAGService
from app.embedding.openai import OpenAIEmbedder
from app.vectorstore.milvus import MilvusStore
from app.llm.openai import OpenAIChatModel

# 返回并缓存RAG服务，在线检索这一条线
@lru_cache
def get_rag_service():
    return RAGService(
        embedder=OpenAIEmbedder(),
        vector_store=MilvusStore(),
        chat_model=OpenAIChatModel()
    )

# 返回并缓存文件处理服务，离线建库
@lru_cache
def get_ingestion_service():
    return IngestionService(
        embedder=OpenAIEmbedder(),
        vector_store=MilvusStore()
    )

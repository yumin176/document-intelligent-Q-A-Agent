"""应用配置模块。

使用 pydantic-settings 读取环境变量和 .env 文件，
避免把密钥、地址等配置硬编码在代码里。
"""

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """全局配置项。"""

    app_name: str = "smart-qa-agent"
    app_version: str = "0.1.0"
    environment: str = "development"
    log_level: str = "INFO"

    # Embedding：默认使用 OpenAI 兼容接口，可用其他服务替换
    embedding_base_url: str = "https://api.openai.com/v1"
    embedding_api_key: str = ""
    embedding_model: str = "text-embedding-3-small"
    embedding_timeout: float = 30.0
    embedding_batch_size: int = 20

    # LLM：默认复用 Embedding 的 base_url / api_key
    llm_base_url: str = ""
    llm_api_key: str = ""
    llm_model: str = "qwen-plus"
    llm_temperature: float = 0.0
    llm_timeout: float = 30.0

    # Rerank API
    rerank_api_url: str = "https://aihubmix.com/v1/rerank"
    rerank_api_key: str = ""
    rerank_model: str = "cohere-rerank-v4.0-fast"
    rerank_timeout: float = 30.0

    # Milvus：本地 Docker Compose 默认地址
    milvus_uri: str = "http://127.0.0.1:19530"
    milvus_collection_name: str = "qa_chunks"
    milvus_timeout: float = 10.0

    # Retrieval
    retrieval_mode: Literal["vector", "bm25", "hybrid", "hybrid_rerank"] = "vector"
    retrieval_candidate_k: int = 10

    # Agent
    agent_max_rewrites: int = 1

    # chunk
    chunk_strategy: Literal["fixed", "recursive"] = "fixed"
    chunk_size: int = 500
    chunk_overlap: int = 50

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    """返回单例配置，避免重复读取文件。"""
    return Settings()

from pydantic import BaseModel
from typing import Any, List

# chat接口入参
class ChatRequest(BaseModel):
    question:str
    top_k:int=4

class SourceOut(BaseModel):
    chunk_id: str
    score: float
    text: str
    metadata: dict
    # 可定位字段：前端引用卡片据此跳转到知识库页面并高亮对应 chunk
    anchor_id: str = ""
    document_id: str = ""
    filename: str = ""
    chunk_index: int = 0
    page_number: int = 0

    @classmethod
    def from_source(cls, source: Any) -> "SourceOut":
        """从 RAG 的 Source 组装响应，定位字段缺失时回退到 chunk_id。"""
        metadata = source.metadata or {}
        return cls(
            chunk_id=source.chunk_id,
            score=source.score,
            text=source.text,
            metadata=metadata,
            anchor_id=str(metadata.get("anchor_id") or source.chunk_id),
            document_id=str(metadata.get("document_id") or ""),
            filename=str(metadata.get("filename") or ""),
            chunk_index=int(metadata.get("chunk_index") or 0),
            page_number=int(metadata.get("page_number") or 0),
        )

# chat接口出参
class ChatResponse(BaseModel):
    answer: str
    sources:List[SourceOut]

# documents接口出参
class DocumentUploadResponse(BaseModel):
    document_id:str
    filename:str
    chunk_count:int
# 知识库查看接口出参
class DocumentSummary(BaseModel):
    document_id: str
    doc_name: str
    filename: str
    file_type: str
    chunk_count: int
    chunk_strategy: str
    chunk_size: int
    chunk_overlap: int
    embedding_model: str
    embedding_dimension: int


class ChunkSummary(BaseModel):
    anchor_id: str
    document_id: str = ""
    chunk_index: int
    page_number: int
    block_type: str
    text: str
    chunk_hash: str

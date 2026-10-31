from pydantic import BaseModel
from typing import List

# chat接口入参
class ChatRequest(BaseModel):
    question:str
    top_k:int=4

class SourceOut(BaseModel):
    chunk_id: str
    score: float
    text: str
    metadata: dict

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
    chunk_index: int
    page_number: int
    block_type: str
    text: str
    chunk_hash: str

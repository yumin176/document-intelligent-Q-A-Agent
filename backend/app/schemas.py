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

"""知识库只读查看接口。"""

from fastapi import APIRouter, Depends, HTTPException

from app.dependencies import get_vector_store
from app.schemas import ChunkSummary, DocumentSummary
from app.vectorstore.base import VectorStore

router = APIRouter(prefix="/api/knowledge", tags=["knowledge"])


@router.get("/documents", response_model=list[DocumentSummary])
def list_documents(
    store: VectorStore = Depends(get_vector_store),
) -> list[DocumentSummary]:
    documents: dict[str, DocumentSummary] = {}
    for record in store.all_chunks():
        metadata = record.metadata
        document_id = metadata.get("document_id")
        if not document_id:
            continue
        if document_id not in documents:
            documents[document_id] = DocumentSummary(
                document_id=document_id,
                doc_name=str(metadata.get("doc_name") or ""),
                filename=str(metadata.get("filename") or ""),
                file_type=str(metadata.get("file_type") or ""),
                chunk_count=0,
                chunk_strategy=str(metadata.get("chunk_strategy") or ""),
                chunk_size=int(metadata.get("chunk_size") or 0),
                chunk_overlap=int(metadata.get("chunk_overlap") or 0),
                embedding_model=str(metadata.get("embedding_model") or ""),
                embedding_dimension=int(metadata.get("embedding_dimension") or 0),
            )
        documents[document_id].chunk_count += 1

    return sorted(
        documents.values(),
        key=lambda item: item.doc_name or item.document_id,
    )


@router.get("/documents/{document_id}/chunks", response_model=list[ChunkSummary])
def list_document_chunks(
    document_id: str,
    store: VectorStore = Depends(get_vector_store),
) -> list[ChunkSummary]:
    chunks = [
        ChunkSummary(
            anchor_id=str(record.id),
            chunk_index=int(record.metadata.get("chunk_index") or 0),
            page_number=int(record.metadata.get("page_number") or 0),
            block_type=str(record.metadata.get("block_type") or "text"),
            text=record.text,
            chunk_hash=str(record.metadata.get("chunk_hash") or ""),
        )
        for record in store.all_chunks()
        if record.metadata.get("document_id") == document_id
    ]
    if not chunks:
        raise HTTPException(status_code=404, detail="文档不存在")
    return sorted(chunks, key=lambda item: item.chunk_index)


@router.get("/chunks", response_model=ChunkSummary)
def get_chunk_by_anchor(
    anchor_id: str,
    store: VectorStore = Depends(get_vector_store),
) -> ChunkSummary:
    for record in store.all_chunks():
        if record.id == anchor_id:
            metadata = record.metadata
            return ChunkSummary(
                anchor_id=str(record.id),
                chunk_index=int(metadata.get("chunk_index") or 0),
                page_number=int(metadata.get("page_number") or 0),
                block_type=str(metadata.get("block_type") or "text"),
                text=record.text,
                chunk_hash=str(metadata.get("chunk_hash") or ""),
            )
    raise HTTPException(status_code=404, detail="锚点不存在")
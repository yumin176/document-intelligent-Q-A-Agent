import unicodedata
from hashlib import sha256
from pathlib import Path

from pydantic import BaseModel

from app.embedding.base import Embedder
from app.ingestion.loader import load_document
from app.ingestion.splitter import split_text
from app.vectorstore.base import ChunkRecord, VectorStore


class IngestResult(BaseModel):
    document_id: str
    filename: str
    chunk_count: int


def normalize_filename(filename: str) -> str:
    """统一大小写和 Unicode，避免 README.md / readme.md 被当成两个文档。"""
    return unicodedata.normalize("NFKC", filename.strip()).casefold()


def build_chunk_records(
    text: str,
    file_meta: dict,
    embedder: Embedder,
    source: str,
) -> list[ChunkRecord]:
    """把文档分块、向量化并组装成可入库记录。"""
    chunks = split_text(text)
    embeddings = embedder.embed_batch(chunks)

    if len(chunks) != len(embeddings):
        raise RuntimeError(
            f"分块数量 {len(chunks)} 与向量数量 {len(embeddings)} 不一致"
        )

    filename = file_meta["filename"]
    identity_key = normalize_filename(filename)
    document_id = sha256(identity_key.encode("utf-8")).hexdigest()[:32]
    content_hash = sha256(text.encode("utf-8")).hexdigest()

    return [
        ChunkRecord(
            id=f"{document_id}:{index}",
            text=chunk,
            embedding=embedding,
            metadata={
                "document_id": document_id,
                "content_hash": content_hash,
                "source": source,
                "filename": file_meta["filename"],
                "file_type": file_meta["file_type"],
                "chunk_index": index,
            },
        )
        for index, (chunk, embedding) in enumerate(zip(chunks, embeddings))
    ]


class IngestionService:
    """离线索引：文档加载 → 分块 → 向量化 → 向量存储。"""

    def __init__(self, embedder: Embedder, vector_store: VectorStore) -> None:
        self.embedder = embedder
        self.vector_store = vector_store

    def ingest_file(
        self,
        path: Path,
        original_filename: str | None = None,
    ) -> IngestResult:
        text, file_meta = load_document(path)

        if original_filename:
            safe_filename = Path(original_filename).name
            file_meta["filename"] = safe_filename
            source = safe_filename
        else:
            source = str(path.resolve())

        records = build_chunk_records(
            text=text,
            file_meta=file_meta,
            embedder=self.embedder,
            source=source,
        )
        self.vector_store.upsert(records)

        document_id = records[0].metadata["document_id"] if records else ""
        return IngestResult(
            document_id=document_id,
            filename=file_meta["filename"],
            chunk_count=len(records),
        )
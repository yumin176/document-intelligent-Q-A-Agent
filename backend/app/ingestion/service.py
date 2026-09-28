from hashlib import sha256

from app.embedding.base import Embedder
from app.ingestion.splitter import split_text
from app.vectorstore.base import ChunkRecord
import unicodedata

def normalize_filename(filename: str) -> str:
    """统一大小写和 Unicode，避免 README.md / readme.md 被当成两个文档。"""
    return unicodedata.normalize("NFKC", filename.strip()).casefold()


def build_chunk_records(
    text: str,
    file_meta: dict,
    embedder: Embedder,
     source: str
) -> list[ChunkRecord]:
    # 文本分块
    chunks = split_text(text)
    # 向量化
    embeddings = embedder.embed_batch(chunks)
    if len(chunks) != len(embeddings):
        raise RuntimeError(
            f"分块数量 {len(chunks)} 与向量数量 {len(embeddings)} 不一致"
        )
    # 确保内容改变后，同一个文本的文档id不会变
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
                "source":source,
                "filename": file_meta["filename"],
                "file_type": file_meta["file_type"],
                "chunk_index": index,
            },
        )
        for index, (chunk, embedding) in enumerate(zip(chunks, embeddings))
    ]

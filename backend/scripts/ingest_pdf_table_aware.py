"""使用 pdfplumber table-aware 解析入库两份年报。"""

import argparse
import sys
from hashlib import sha256
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.config import get_settings
from app.embedding.fake import FakeEmbedder
from app.embedding.openai import OpenAIEmbedder
from app.ingestion.pdf_table_aware import parse_table_aware_pdf, split_table_aware_blocks
from app.ingestion.service import build_anchor_id, normalize_doc_name
from app.vectorstore.base import ChunkRecord
from app.vectorstore.memory import InMemoryVectorStore
from app.vectorstore.milvus import MilvusStore

PILOT_PDFS = (
    "上市公司年报/极米科技_2025年年度报告.pdf",
    "上市公司年报/科大讯飞_2025年年度报告.pdf",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="table-aware PDF 入库")
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--collection", default="qa_chunks_pilot_tableaware")
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--chunk-overlap", type=int, default=50)
    parser.add_argument("--strategy", choices=("fixed", "recursive"), default="fixed")
    parser.add_argument("--docs", nargs="+", default=list(PILOT_PDFS))
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def ingest_pdf(
    path: Path,
    *,
    doc_name: str,
    embedder,
    store,
    chunk_size: int,
    chunk_overlap: int,
    strategy: str,
) -> dict:
    blocks = parse_table_aware_pdf(path)
    chunks = split_table_aware_blocks(
        blocks,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        strategy=strategy,
    )
    if not chunks:
        raise ValueError(f"没有可入库内容: {doc_name}")

    texts = [chunk.text for chunk in chunks]
    embeddings = embedder.embed_batch(texts)
    embedding_dimension = len(embeddings[0]) if embeddings else 0
    embedding_model = getattr(embedder, "model", get_settings().embedding_model)
    embedding_base_url = getattr(embedder, "base_url", get_settings().embedding_base_url)
    if len(texts) != len(embeddings):
        raise RuntimeError("chunk 数量与 embedding 数量不一致")

    filename = path.name
    document_id = sha256(
        normalize_doc_name(doc_name).encode("utf-8")
    ).hexdigest()[:32]
    raw_content = "\n".join(block.text for block in blocks)
    content_hash = sha256(raw_content.encode("utf-8")).hexdigest()
    page_count = max((block.page_number for block in blocks), default=0)

    records: list[ChunkRecord] = []
    for index, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        anchor_id = build_anchor_id(document_id, index, chunk.text)
        records.append(
            ChunkRecord(
                id=anchor_id,
                text=chunk.text,
                embedding=embedding,
                metadata={
                    "anchor_id": anchor_id,
                    "doc_name": doc_name,
                    "chunk_hash": sha256(chunk.text.encode("utf-8")).hexdigest(),
                    "document_id": document_id,
                    "content_hash": content_hash,
                    "source": doc_name,
                    "filename": filename,
                    "file_type": "pdf",
                    "chunk_index": index,
                    "raw_num_chars": len(raw_content),
                    "num_chars": sum(len(text) for text in texts),
                    "cleaning_steps": [
                        "pdf_table_aware",
                        "pdf_page_noise",
                        "normalize",
                        "media",
                        "noise",
                        "whitespace",
                        "dedup",
                    ],
                    "page_cleaning": {
                        "enabled": True,
                        "parser": "pdfplumber_table_aware",
                    },
                    "chunk_strategy": f"table_aware_{strategy}",
                    "chunk_size": chunk_size,
                    "chunk_overlap": chunk_overlap,
                    "embedding_model": embedding_model,
                    "embedding_base_url": embedding_base_url,
                    "embedding_dimension": embedding_dimension,
                    "block_type": chunk.block_type,
                    "page_number": chunk.page_number,
                    **chunk.metadata,
                },
            )
        )

    store.upsert(records)
    return {
        "doc_name": doc_name,
        "document_id": document_id,
        "page_count": page_count,
        "text_chunks": sum(chunk.block_type == "text" for chunk in chunks),
        "table_chunks": sum(chunk.block_type == "table" for chunk in chunks),
        "chunk_count": len(records),
    }


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

    args = parse_args()
    if args.chunk_size <= 0 or not 0 <= args.chunk_overlap < args.chunk_size:
        print("分块参数不合法", file=sys.stderr)
        return 2

    paths = [(doc_name, args.corpus_root / doc_name) for doc_name in args.docs]
    missing = [str(path) for _, path in paths if not path.exists()]
    if missing:
        print("以下文件不存在：", file=sys.stderr)
        for path in missing:
            print(f"- {path}", file=sys.stderr)
        return 2

    if args.dry_run:
        embedder = FakeEmbedder(dim=8)
        store = InMemoryVectorStore()
        mode = "dry-run"
    else:
        embedder = OpenAIEmbedder()
        store = MilvusStore(collection_name=args.collection)
        mode = args.collection

    print(
        f"mode={mode} parser=pdfplumber_table_aware strategy={args.strategy} "
        f"chunk_size={args.chunk_size} chunk_overlap={args.chunk_overlap}"
    )
    total = 0
    try:
        for doc_name, path in paths:
            summary = ingest_pdf(
                path,
                doc_name=doc_name,
                embedder=embedder,
                store=store,
                chunk_size=args.chunk_size,
                chunk_overlap=args.chunk_overlap,
                strategy=args.strategy,
            )
            total += summary["chunk_count"]
            print(
                f"{doc_name} -> document_id={summary['document_id']} "
                f"pages={summary['page_count']} "
                f"text_chunks={summary['text_chunks']} "
                f"table_chunks={summary['table_chunks']} "
                f"chunks={summary['chunk_count']}"
            )
    finally:
        if not args.dry_run:
            store.close()

    print(f"documents={len(paths)} chunks={total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
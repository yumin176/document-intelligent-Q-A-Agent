"""批量入库语料文件到指定 Milvus Collection，并生成 manifest。"""

import argparse
import sys
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.config import get_settings
from app.embedding.fake import FakeEmbedder
from app.embedding.openai import OpenAIEmbedder
from app.ingestion.manifest import save_manifest
from app.ingestion.service import IngestionService
from app.vectorstore.memory import InMemoryVectorStore
from app.vectorstore.milvus import MilvusStore

PILOT_DOCS = (
    "上市公司年报/极米科技_2025年年度报告.pdf",
    "上市公司年报/科大讯飞_2025年年度报告.pdf",
    "技术文档-Vue3中文/guide/essentials/computed.md",
    "技术文档-Vue3中文/guide/components/props.md",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="批量入库语料")
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--collection", default="qa_chunks_pilot")
    parser.add_argument(
        "--strategy",
        choices=("fixed", "recursive"),
        default="recursive",
    )
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--chunk-overlap", type=int, default=50)
    parser.add_argument(
        "--docs",
        nargs="+",
        default=list(PILOT_DOCS),
        help="相对 corpus-root 的文档路径；默认使用 4 份 pilot 文档",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="使用 FakeEmbedder + InMemoryVectorStore，不连接 Milvus",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=BACKEND_ROOT / "evaluation" / "corpus_manifest.json",
        help="ingestion manifest 输出路径",
    )
    args = parser.parse_args(argv)
    if args.chunk_size <= 0:
        parser.error("--chunk-size 必须大于 0")
    if not 0 <= args.chunk_overlap < args.chunk_size:
        parser.error("--chunk-overlap 必须满足 0 <= overlap < chunk-size")
    if len(set(args.docs)) != len(args.docs):
        parser.error("--docs 中存在重复路径")
    return args


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

    args = parse_args(argv)
    paths = [(doc_name, args.corpus_root / doc_name) for doc_name in args.docs]
    missing = [str(path) for _, path in paths if not path.exists()]
    if missing:
        print("以下文档不存在：", file=sys.stderr)
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

    service = IngestionService(
        embedder=embedder,
        vector_store=store,
        chunk_strategy=args.strategy,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )

    print(
        f"mode={mode} strategy={args.strategy} "
        f"chunk_size={args.chunk_size} chunk_overlap={args.chunk_overlap}"
    )
    total_chunks = 0
    documents = []
    try:
        for doc_name, path in paths:
            result = service.ingest_file(path, doc_name=doc_name)
            total_chunks += result.chunk_count
            documents.append(
                {
                    "doc_name": doc_name,
                    "document_id": result.document_id,
                    "chunk_count": result.chunk_count,
                    "content_hash": sha256(path.read_bytes()).hexdigest(),
                    "ingested_at": datetime.now(timezone.utc).isoformat(),
                }
            )
            print(
                f"{doc_name} -> document_id={result.document_id} "
                f"chunks={result.chunk_count}"
            )

        records = store.all_chunks()
        first_metadata = records[0].metadata if records else {}
        manifest = save_manifest(
            args.manifest,
            collection_name=args.collection if not args.dry_run else "dry-run",
            embedding_model=str(
                first_metadata.get("embedding_model")
                or getattr(embedder, "model", get_settings().embedding_model)
            ),
            embedding_base_url=str(
                first_metadata.get("embedding_base_url")
                or getattr(embedder, "base_url", get_settings().embedding_base_url)
            ),
            embedding_dimension=int(first_metadata.get("embedding_dimension") or 0),
            chunk_strategy=args.strategy,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            documents=documents,
        )
        print(f"manifest={args.manifest}")
    finally:
        if not args.dry_run:
            store.close()

    print(f"documents={len(paths)} chunks={total_chunks}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
"""批量入库语料文件到指定 Milvus Collection。"""

import argparse
import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.embedding.fake import FakeEmbedder
from app.embedding.openai import OpenAIEmbedder
from app.ingestion.service import IngestionService
from app.vectorstore.memory import InMemoryVectorStore
from app.vectorstore.milvus import MilvusStore

PILOT_DOCS = (
    "上市公司年报/极米科技_2025年年度报告.pdf",
    "上市公司年报/科大讯飞_2025年年度报告.pdf",
    "技术文档-Vue3中文/guide/essentials/computed.md",
    "技术文档-Vue3中文/guide/components/props.md",
)


def parse_args() -> argparse.Namespace:
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
    args = parser.parse_args()
    if args.chunk_size <= 0:
        parser.error("--chunk-size 必须大于 0")
    if not 0 <= args.chunk_overlap < args.chunk_size:
        parser.error("--chunk-overlap 必须满足 0 <= overlap < chunk-size")
    if len(set(args.docs)) != len(args.docs):
        parser.error("--docs 中存在重复路径")
    return args


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

    args = parse_args()
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
    try:
        for doc_name, path in paths:
            result = service.ingest_file(path, doc_name=doc_name)
            total_chunks += result.chunk_count
            print(
                f"{doc_name} -> document_id={result.document_id} "
                f"chunks={result.chunk_count}"
            )
    finally:
        if not args.dry_run:
            store.close()

    print(f"documents={len(paths)} chunks={total_chunks}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
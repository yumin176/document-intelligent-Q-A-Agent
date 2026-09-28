"""把 js.md 文档解析、分块、向量化后写入 Milvus，并做一次检索演示。"""

import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.embedding.openai import OpenAIEmbedder
from app.ingestion.loader import load_document
from app.ingestion.service import build_chunk_records
from app.vectorstore.milvus import MilvusStore

FILE_PATH = "E:/01-学习/02-文档笔记/js.md"
QUERY = "const和let有什么区别"


def main() -> None:
    file_path = Path(FILE_PATH)

    print("-" * 20, "正在读取文件", "-" * 20)
    text, file_meta = load_document(file_path)

    embedder = OpenAIEmbedder()

    print("-" * 20, "正在分块并向量化", "-" * 20)
    records = build_chunk_records(
        text=text,
        file_meta=file_meta,
        embedder=embedder,
        source=str(file_path.resolve()),
    )
    print(f"共生成 {len(records)} 个 chunk")

    store = MilvusStore()
    try:
        print("-" * 20, "正在写入 Milvus", "-" * 20)
        store.upsert(records)

        print("-" * 20, f"查询问题: {QUERY}", "-" * 20)
        query_embedding = embedder.embed(QUERY)
        results = store.search(vector=query_embedding, top_k=3)

        print("-" * 20, "查询结果", "-" * 20)
        for result in results:
            print(f"\nscore = {result.score:.4f}")
            print(f"id = {result.id}")
            print(f"source = {result.metadata.get('source')}")
            print(f"chunk_index = {result.metadata.get('chunk_index')}")
            print(result.text[:300])
    finally:
        store.close()


if __name__ == "__main__":
    main()
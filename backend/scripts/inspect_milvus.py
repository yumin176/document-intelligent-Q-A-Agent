"""只读查看本地 Milvus 中的数据。

用法：
    python scripts/inspect_milvus.py
    python scripts/inspect_milvus.py --collection qa_chunks --limit 5
    python scripts/inspect_milvus.py --collection qa_chunks --limit 3 --show-vector
"""

import argparse
import json
import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from pymilvus import MilvusClient

from app.config import get_settings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="查看 Milvus Collection 数据")
    parser.add_argument("--collection", default=None, help="Collection 名称")
    parser.add_argument("--limit", type=int, default=10, help="最多显示多少条")
    parser.add_argument("--filter", default="", help="Milvus 过滤表达式，例如 id == \"0\"")
    parser.add_argument("--show-vector", action="store_true", help="显示向量维度和前 5 个值")
    return parser.parse_args()


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    args = parse_args()
    settings = get_settings()
    collection_name = args.collection or settings.milvus_collection_name

    client = MilvusClient(
        uri=settings.milvus_uri,
        timeout=settings.milvus_timeout,
    )

    try:
        collections = client.list_collections()
        print(f"Collections: {collections}")

        if collection_name not in collections:
            print(f"Collection 不存在: {collection_name}")
            return

        stats = client.get_collection_stats(collection_name)
        print(f"Collection: {collection_name}")
        print(f"Row count: {stats.get('row_count', 'unknown')}")

        output_fields = ["id", "text", "metadata_json"]
        if args.show_vector:
            output_fields.append("vector")

        rows = client.query(
            collection_name=collection_name,
            filter=args.filter,
            output_fields=output_fields,
            limit=max(args.limit, 1),
            consistency_level="Strong",
        )

        rows.sort(
            key=lambda row: (
                0,
                int(str(row.get("id"))),
            )
            if str(row.get("id", "")).isdigit()
            else (1, str(row.get("id", "")))
        )

        if not rows:
            print("没有查询到数据。")
            return

        for index, row in enumerate(rows, start=1):
            metadata = row.get("metadata_json") or "{}"
            try:
                metadata_value = json.loads(metadata)
            except json.JSONDecodeError:
                metadata_value = metadata

            print("\n" + "=" * 70)
            print(f"[{index}] id = {row.get('id')}")
            print("text:")
            print(row.get("text") or "")
            print("metadata:")
            print(json.dumps(metadata_value, ensure_ascii=False, indent=2))

            if args.show_vector:
                vector = row.get("vector") or []
                print(f"vector dimension: {len(vector)}")
                print(f"vector preview: {vector[:5]}")
    finally:
        client.close()


if __name__ == "__main__":
    main()
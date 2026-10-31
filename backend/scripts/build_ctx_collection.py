"""给已有 collection 的 chunk 注入实体上下文前缀，重建一个新 collection。

为什么只加前缀、不重新解析
--------------------------
年报里"主要会计数据"等表格在 5 家公司中措辞几乎完全同构，
而 chunk 内部不含公司名，实体只存在于 query 里，导致
"问科大讯飞、返回极米净利润"这类跨文档混淆。
本脚本保持分块边界完全不变，只在块首插入一行
`【公司｜年份｜第 N 页｜表 M】`，从而把"分块策略"和
"实体上下文"两个变量彻底解耦，便于 A/B 对照。

用法
----
    python scripts/build_ctx_collection.py \
        --source qa_chunks_pdf5_fixed --target qa_chunks_pdf5_ctx
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from hashlib import sha256
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.embedding.openai import OpenAIEmbedder
from app.vectorstore.base import ChunkRecord, TextRecord
from app.vectorstore.milvus import MilvusStore

# 年报文件名 → (公司简称, 股票代码)
COMPANY_CODES = {
    "科大讯飞": "002230",
    "极米科技": "688696",
    "成都银行": "601838",
    "通威股份": "600438",
    "纵横股份": "688070",
    "新易盛": "300502",
    "五粮液": "000858",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="重建带实体上下文的 collection")
    parser.add_argument("--source", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--dry-run", action="store_true", help="只打印前缀样例")
    return parser.parse_args()


def parse_doc_identity(doc_name: str) -> tuple[str, str]:
    """从 `上市公司年报/极米科技_2025年年度报告.pdf` 取出公司名与年份。"""
    stem = Path(doc_name.replace("\\", "/")).stem
    match = re.match(r"^(?P<company>[^_]+)_(?P<year>\d{4})年", stem)
    if match:
        return match.group("company"), match.group("year")
    return stem, ""


def build_header(chunk: TextRecord) -> str:
    metadata = chunk.metadata
    document = metadata.get("doc_name") or metadata.get("filename") or ""
    company, year = parse_doc_identity(document)
    code = COMPANY_CODES.get(company, "")

    parts = [f"{company}({code})" if code else company]
    if year:
        parts.append(f"{year}年年度报告")
    page = metadata.get("page_number")
    if page:
        parts.append(f"第{page}页")
    if metadata.get("block_type") == "table":
        table_index = metadata.get("table_index")
        parts.append(f"表{table_index}" if table_index else "表格")
    return "【" + "｜".join(parts) + "】"


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args()

    # 注意：pymilvus 的 close() 会断开共享连接别名，
    # 两个 client 同时存活时不能在中途关闭其中一个。
    source = MilvusStore(collection_name=args.source)
    chunks = source.all_chunks()
    target = MilvusStore(collection_name=args.target)
    if not chunks:
        print(f"源 collection 为空: {args.source}", file=sys.stderr)
        return 1

    prefixed = [(chunk, build_header(chunk) + chunk.text) for chunk in chunks]
    print(f"source={args.source} chunks={len(chunks)}")
    print("前缀样例：")
    for chunk, text in prefixed[:3]:
        print(f"  {text[:80]!r}")
    if args.dry_run:
        return 0

    embedder = OpenAIEmbedder()
    total_written = 0
    try:
        for start in range(0, len(prefixed), args.batch_size):
            batch = prefixed[start : start + args.batch_size]
            embeddings = embedder.embed_batch([text for _, text in batch])
            records = [
                ChunkRecord(
                    id=chunk.id,
                    text=text,
                    embedding=embedding,
                    metadata={
                        **chunk.metadata,
                        "header_context": True,
                        "raw_chunk_hash": chunk.metadata.get("chunk_hash")
                        or sha256(chunk.text.encode("utf-8")).hexdigest(),
                    },
                )
                for (chunk, text), embedding in zip(batch, embeddings)
            ]
            target.upsert(records)
            total_written += len(records)
            print(f"  {total_written}/{len(prefixed)}", end="\r", flush=True)
    finally:
        target.close()
        source.close()

    print(f"\n完成: {args.target} 写入 {total_written} 块")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""基准测试：all_chunks() 的 Strong vs Bounded 一致性。

复测方式：cd backend && .venv/Scripts/python.exe scripts/bench_all_chunks.py
"""

import json
import time

from pymilvus import MilvusClient

URI = "http://127.0.0.1:19530"
COLLECTION = "qa_chunks_prod_v3"
N_RUNS = 5

client = MilvusClient(uri=URI, timeout=10.0)
assert client.has_collection(COLLECTION), f"集合 {COLLECTION} 不存在"

stats = client.get_collection_stats(COLLECTION)
print(f"集合: {COLLECTION}  行数: {stats.get('row_count')}")


def bench(label: str, fn) -> None:
    times = []
    n_rows = 0
    for _ in range(N_RUNS):
        start = time.perf_counter()
        rows = fn()
        times.append((time.perf_counter() - start) * 1000)
        n_rows = len(rows)
    avg = sum(times) / len(times)
    best = min(times)
    print(f"{label:<44} avg={avg:7.1f}ms  best={best:7.1f}ms  rows={n_rows}")


full_fields = ["id", "text", "metadata_json"]
meta_only = ["id", "metadata_json"]

# 1. Strong 全字段（当前实现）
bench("Strong + 全字段 (现状)",
      lambda: client.query(collection_name=COLLECTION, filter="",
                           output_fields=full_fields, limit=16384,
                           consistency_level="Strong"))

# 2. Bounded 全字段（get_by_ids 同款）
bench("Bounded + 全字段",
      lambda: client.query(collection_name=COLLECTION, filter="",
                           output_fields=full_fields, limit=16384,
                           consistency_level="Bounded"))

# 3. Bounded + 只要 id/metadata（/documents 列表场景不需要 text）
bench("Bounded + 仅 id/metadata_json",
      lambda: client.query(collection_name=COLLECTION, filter="",
                           output_fields=meta_only, limit=16384,
                           consistency_level="Bounded"))

# 4. Eventual + 仅 id/metadata
bench("Eventual + 仅 id/metadata_json",
      lambda: client.query(collection_name=COLLECTION, filter="",
                           output_fields=meta_only, limit=16384,
                           consistency_level="Eventually"))

# 看看 text 占多大比例
rows = client.query(collection_name=COLLECTION, filter="", output_fields=full_fields,
                    limit=16384, consistency_level="Bounded")
text_bytes = sum(len(r.get("text", "").encode("utf-8")) for r in rows)
meta_bytes = sum(len(r.get("metadata_json", "").encode("utf-8")) for r in rows)
print(f"\n载荷分析: text={text_bytes/1024:.0f}KB  metadata={meta_bytes/1024:.0f}KB  "
      f"text占比={text_bytes/(text_bytes+meta_bytes)*100:.0f}%")

# 文档数量（/documents 真正需要的信息）
doc_ids = set()
for r in rows:
    meta = json.loads(r.get("metadata_json") or "{}")
    if meta.get("document_id"):
        doc_ids.add(meta["document_id"])
print(f"文档数: {len(doc_ids)}  chunk总数: {len(rows)}")

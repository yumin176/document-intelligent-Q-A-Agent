"""精排探针 V3：一次取候选，离线比较多种精排融合策略。

- pure：完全按 rerank 分数重排
- wX：RRF 分数与 rerank 分数线性融合（X 为 rerank 权重）
- tail：保留原序前 head 名，只重排其余位置
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import get_settings  # noqa: E402
from app.dependencies import get_retriever  # noqa: E402
from app.evaluation.retrieval import normalize_match_text  # noqa: E402

RERANK_URL = "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank"
HEADER_RE = re.compile(r"^【[^】]*】\s*")


def minmax(values: list[float]) -> list[float]:
    low, high = min(values), max(values)
    if high - low < 1e-9:
        return [0.5] * len(values)
    return [(v - low) / (high - low) for v in values]


def call_rerank(client: httpx.Client, key: str, model: str, query: str, docs: list[str]) -> list[float]:
    payload = {
        "model": model,
        "input": {"query": query, "documents": docs},
        "parameters": {"top_n": len(docs)},
    }
    response = client.post(
        RERANK_URL,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json=payload,
    )
    response.raise_for_status()
    scores = [0.0] * len(docs)
    for item in response.json()["output"]["results"]:
        scores[item["index"]] = float(item["relevance_score"])
    return scores


def hit_at_k(case: dict, texts: list[str], k: int) -> tuple[bool, int | None]:
    snippets = [normalize_match_text(s) for s in case["relevant_snippets"]]
    for rank, text in enumerate(texts[:k], start=1):
        normalized = normalize_match_text(text)
        if all(s in normalized for s in snippets):
            return True, rank
    return False, None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--pool", type=int, default=10)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--model", default="gte-rerank-v2")
    parser.add_argument("--head", type=int, default=4, help="tail 策略保留不动的名次数")
    parser.add_argument(
        "--strip-header",
        action="store_true",
        help="送精排前去掉块首的【公司｜年份】上下文前缀，用于验证前缀是否压缩了打分区分度",
    )
    parser.add_argument("--output")
    args = parser.parse_args()

    key = get_settings().embedding_api_key
    cases = [json.loads(l) for l in Path(args.dataset).read_text("utf-8").splitlines() if l.strip()]
    retriever = get_retriever()

    strategies = ["baseline", "pure"] + [f"w{w}" for w in (0.2, 0.4, 0.6, 0.8)] + ["tail"]
    stats = {name: {"hit": 0, "rr": 0.0} for name in strategies}
    per_case: list[dict] = []

    with httpx.Client(timeout=60, trust_env=True) as client:
        for case in cases:
            hits = retriever.search(case["question"], top_k=args.pool)
            texts = [h.text for h in hits]
            fusion = minmax([h.score for h in hits])
            if args.strip_header:
                payload_docs = [HEADER_RE.sub("", t, count=1).strip() for t in texts]
            else:
                payload_docs = texts
            rr = call_rerank(client, key, args.model, case["question"], payload_docs)
            rr_norm = minmax(rr)
            orderings = {"baseline": list(range(len(texts))), "pure": sorted(range(len(texts)), key=lambda i: rr[i], reverse=True)}
            for w in (0.2, 0.4, 0.6, 0.8):
                orderings[f"w{w}"] = sorted(
                    range(len(texts)), key=lambda i: w * rr_norm[i] + (1 - w) * fusion[i], reverse=True
                )
            head = list(range(min(args.head, len(texts))))
            tail = sorted(range(len(texts))[len(head):], key=lambda i: rr[i], reverse=True)
            orderings["tail"] = head + tail

            row = {"case_id": case["id"]}
            for name, order in orderings.items():
                ok, rank = hit_at_k(case, [texts[i] for i in order], args.top_k)
                stats[name]["hit"] += ok
                if rank:
                    stats[name]["rr"] += 1 / rank
                row[name] = ok
            per_case.append(row)

    total = len(cases)
    print(f"样本 {total} | pool={args.pool} top_k={args.top_k} model={args.model}")
    print(f"{'策略':10s}{'Recall':>8s}{'MRR':>8s}")
    for name in strategies:
        print(f"{name:10s}{stats[name]['hit'] / total:8.3f}{stats[name]['rr'] / total:8.3f}")

    print("\n逐题对比（纯精排 vs 基线）:")
    for row in per_case:
        if row["baseline"] != row["pure"]:
            print(f"  {row['case_id']:16s} baseline={row['baseline']} pure={row['pure']} w0.4={row['w0.4']} tail={row['tail']}")

    if args.output:
        Path(args.output).write_text(
            json.dumps(
                {
                    "dataset": str(args.dataset),
                    "model": args.model,
                    "pool": args.pool,
                    "top_k": args.top_k,
                    "case_count": total,
                    "summary": {k: {"recall": v["hit"] / total, "mrr": v["rr"] / total} for k, v in stats.items()},
                    "cases": per_case,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

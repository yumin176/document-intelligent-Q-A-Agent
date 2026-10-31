"""离线融合策略扫描：候选只取一次，融合参数在内存里全网格对比。

动机
----
线上 `HybridRetriever` 用的是等权 RRF（rrf_k=60），但 pdf5 实测：
单路 BM25 0.56 > hybrid 0.52，扩池后 hybrid 还掉到 0.44。
说明"两路共识"在年报这类同构表格语料上会压制真正的答案块。
本脚本把 BM25 / 向量候选各取一次（深池），然后在内存里穷举
rrf_k 与加权系数，直接输出最优区间，不重复消耗 embedding 配额。

用法
----
    python scripts/tune_fusion.py --collection qa_chunks_pdf5_fixed \
        --dataset evaluation/retrieval_cases_pdf.jsonl --pool 100
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

import numpy as np
from rank_bm25 import BM25Okapi

from app.embedding.openai import OpenAIEmbedder
from app.evaluation.retrieval import (
    RetrievalCase,
    hit_at_k,
    load_cases,
    matched_snippet_indexes,
    mrr_at_k,
    ndcg_at_k,
    recall_at_k,
)
from app.retrieval.bm25 import tokenize
from app.vectorstore.base import SearchResult, TextRecord
from app.vectorstore.milvus import MilvusStore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="融合策略离线扫描")
    parser.add_argument("--collection", required=True)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=BACKEND_ROOT / "evaluation" / "retrieval_cases_pdf.jsonl",
    )
    parser.add_argument("--pool", type=int, default=100, help="每路候选深度")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--rrf-ks",
        type=int,
        nargs="+",
        default=[1, 3, 5, 10, 20, 60],
        help="要扫描的 rrf_k 取值",
    )
    parser.add_argument(
        "--weights",
        type=float,
        nargs="+",
        default=[0.0, 0.3, 0.5, 0.7, 1.0],
        help="要扫描的 BM25 侧权重",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if args.pool <= 0 or args.top_k <= 0:
        parser.error("--pool 与 --top-k 必须大于 0")
    return args


def build_bm25(chunks: list[TextRecord]) -> BM25Okapi:
    return BM25Okapi([tokenize(chunk.text) for chunk in chunks])


def rank_bm25(bm25: BM25Okapi, question: str, pool: int) -> list[int]:
    scores = bm25.get_scores(tokenize(question))
    order = np.argsort(scores)[::-1][:pool]
    return [int(index) for index in order if scores[index] > 0]


def rank_vector(
    embedder: OpenAIEmbedder,
    store: MilvusStore,
    chunks: list[TextRecord],
    index_by_id: dict[str, int],
    question: str,
    pool: int,
) -> list[int]:
    hits = store.search(embedder.embed(question), top_k=pool)
    return [index_by_id[hit.id] for hit in hits if hit.id in index_by_id]


def fuse_rrf(
    bm25_rank: list[int],
    vector_rank: list[int],
    pool: int,
    *,
    rrf_k: int,
    weight_bm25: float = 0.5,
) -> list[int]:
    """加权 RRF：weight_bm25 为 BM25 侧权重，向量侧取 1-w。"""
    scores: dict[int, float] = {}
    weight_vector = 1.0 - weight_bm25
    for weight, ranking in ((weight_bm25, bm25_rank), (weight_vector, vector_rank)):
        if weight <= 0:
            continue
        for rank, index in enumerate(ranking, start=1):
            scores[index] = scores.get(index, 0.0) + weight / (rrf_k + rank)
    ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    return [index for index, _ in ordered][:pool]


def evaluate_ranking(
    ranking: list[int],
    chunks: list[TextRecord],
    case: RetrievalCase,
    top_k: int,
) -> tuple[float, float, float, float]:
    matches: list[set[int]] = []
    for index in ranking[:top_k]:
        chunk = chunks[index]
        matches.append(matched_snippet_indexes(SearchResult(
            id=chunk.id,
            score=0.0,
            text=chunk.text,
            metadata=chunk.metadata,
        ), case))
    snippet_count = len(case.relevant_snippets)
    return (
        recall_at_k(matches, snippet_count, top_k),
        hit_at_k(matches, top_k),
        mrr_at_k(matches, top_k),
        ndcg_at_k(matches, snippet_count, top_k),
    )


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args()

    cases = load_cases(args.dataset)
    store = MilvusStore(collection_name=args.collection)
    embedder = OpenAIEmbedder()
    try:
        chunks = store.all_chunks()
        if not chunks:
            print(f"collection 为空: {args.collection}", file=sys.stderr)
            return 1
        index_by_id = {chunk.id: index for index, chunk in enumerate(chunks)}
        print(f"collection={args.collection} chunks={len(chunks)} cases={len(cases)}")

        bm25 = build_bm25(chunks)
        bm25_ranks: dict[str, list[int]] = {}
        vector_ranks: dict[str, list[int]] = {}
        for case in cases:
            bm25_ranks[case.case_id] = rank_bm25(bm25, case.question, args.pool)
            vector_ranks[case.case_id] = rank_vector(
                embedder, store, chunks, index_by_id, case.question, args.pool
            )
    finally:
        store.close()

    strategies: dict[str, dict[str, list[int]]] = {}
    for rrf_k in args.rrf_ks:
        for weight in args.weights:
            if rrf_k == 60 and weight == 0.5:
                label = "rrf_k=60 w=0.5(线上现状)"
            else:
                label = f"rrf_k={rrf_k} w={weight:.2f}"
            strategies[label] = {
                case.case_id: fuse_rrf(
                    bm25_ranks[case.case_id],
                    vector_ranks[case.case_id],
                    args.pool,
                    rrf_k=rrf_k,
                    weight_bm25=weight,
                )
                for case in cases
            }

    rows = []
    for label, rankings in strategies.items():
        totals = [0.0, 0.0, 0.0, 0.0]
        for case in cases:
            metrics = evaluate_ranking(rankings[case.case_id], chunks, case, args.top_k)
            for index, value in enumerate(metrics):
                totals[index] += value
        rows.append((label, *[total / len(cases) for total in totals]))

    rows.sort(key=lambda row: (-row[1], -row[4]))
    print(f"\n{'策略':28s}{'Recall@'+str(args.top_k):>12s}{'Hit':>8s}{'MRR':>8s}{'nDCG':>8s}")
    for label, recall, hit, mrr, ndcg in rows:
        print(f"{label:28s}{recall:12.4f}{hit:8.3f}{mrr:8.3f}{ndcg:8.3f}")

    if args.output:
        best = rows[0][0]
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(
                {
                    "collection": args.collection,
                    "dataset": str(args.dataset),
                    "pool": args.pool,
                    "top_k": args.top_k,
                    "scores": [
                        {"strategy": row[0], "recall": row[1], "hit": row[2],
                         "mrr": row[3], "ndcg": row[4]}
                        for row in rows
                    ],
                    "best": best,
                    "best_per_case": {
                        case.case_id: [
                            chunks[index].metadata.get("doc_name", "").split("/")[-1]
                            for index in strategies[best][case.case_id][: args.top_k]
                        ]
                        for case in cases
                    },
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"\n已写入 {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

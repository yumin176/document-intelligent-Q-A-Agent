"""对比评测向量、BM25、混合检索和 Rerank 检索。

用法：
    python scripts/evaluate_retrieval.py
    python scripts/evaluate_retrieval.py --top-k 5 --candidate-k 10
    python scripts/evaluate_retrieval.py --modes vector bm25 hybrid hybrid_rerank
    python scripts/evaluate_retrieval.py --show-misses
    python scripts/evaluate_retrieval.py --output evaluation/results.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.config import get_settings
from app.embedding.openai import OpenAIEmbedder
from app.evaluation.retrieval import (
    ModeEvaluation,
    evaluate_retriever,
    load_cases,
)
from app.rerank.cohere import CohereReranker
from app.retrieval.base import Retriever
from app.retrieval.bm25 import BM25Retriever
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.reranked import RerankedRetriever
from app.retrieval.vector import VectorRetriever
from app.vectorstore.milvus import MilvusStore

MODES = ("vector", "bm25", "hybrid", "hybrid_rerank")


def parse_args(settings) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="检索模式对比评测")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=BACKEND_ROOT / "evaluation" / "retrieval_cases.jsonl",
        help="评测集路径（JSONL 或 JSON 数组）",
    )
    parser.add_argument("--top-k", type=int, default=5, help="最终返回条数")
    parser.add_argument(
        "--candidate-k",
        type=int,
        default=settings.retrieval_candidate_k,
        help="召回候选条数",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=MODES,
        default=list(MODES),
        help="要对比的检索模式",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="可选：把完整结果写入 JSON 文件",
    )
    parser.add_argument(
        "--show-misses",
        action="store_true",
        help="打印未完全召回相关 chunk 的问题",
    )
    args = parser.parse_args()
    if args.top_k <= 0:
        parser.error("--top-k 必须大于 0")
    if args.candidate_k <= 0:
        parser.error("--candidate-k 必须大于 0")
    return args


def build_retriever(
    mode: str,
    embedder: OpenAIEmbedder,
    store: MilvusStore,
    candidate_k: int,
) -> tuple[Retriever, CohereReranker | None]:
    vector_retriever = VectorRetriever(embedder, store)

    if mode == "vector":
        return vector_retriever, None

    bm25_retriever = BM25Retriever(store)
    if mode == "bm25":
        return bm25_retriever, None

    hybrid_retriever = HybridRetriever(
        vector_retriever=vector_retriever,
        bm25_retriever=bm25_retriever,
        candidate_k=candidate_k,
    )
    if mode == "hybrid":
        return hybrid_retriever, None

    reranker = CohereReranker(max_candidates=candidate_k)
    return (
        RerankedRetriever(
            base_retriever=hybrid_retriever,
            reranker=reranker,
            candidate_k=candidate_k,
        ),
        reranker,
    )


def print_summary(mode: str, evaluation: ModeEvaluation, top_k: int) -> None:
    summary = evaluation.summary
    print(
        f"{mode:<16}"
        f"Recall@{top_k}={summary['recall']:.4f}  "
        f"Hit@{top_k}={summary['hit']:.4f}  "
        f"MRR@{top_k}={summary['mrr']:.4f}  "
        f"nDCG@{top_k}={summary['ndcg']:.4f}  "
        f"latency={summary['latency_ms']:.2f}ms"
    )


def print_misses(evaluation: ModeEvaluation) -> None:
    misses = [query for query in evaluation.queries if query.recall < 1.0]
    if not misses:
        return
    print(f"\n{evaluation.mode} 未完全召回：")
    for query in misses:
        print(
            f"- {query.case_id}: {query.question}\n"
            f"  relevant={query.relevant_keys} "
            f"retrieved={query.retrieved_keys} "
            f"rank={query.first_relevant_rank}"
        )


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

    settings = get_settings()
    args = parse_args(settings)
    cases = load_cases(args.dataset)

    print(f"评测集: {args.dataset}")
    print(f"样本数: {len(cases)} | top_k={args.top_k} | candidate_k={args.candidate_k}")
    print("-" * 100)

    embedder = OpenAIEmbedder()
    store = MilvusStore()
    results: dict[str, ModeEvaluation] = {}
    errors: dict[str, str] = {}

    try:
        for mode in args.modes:
            reranker: CohereReranker | None = None
            try:
                retriever, reranker = build_retriever(
                    mode=mode,
                    embedder=embedder,
                    store=store,
                    candidate_k=args.candidate_k,
                )
                evaluation = evaluate_retriever(
                    retriever=retriever,
                    cases=cases,
                    top_k=args.top_k,
                    mode=mode,
                )
                results[mode] = evaluation
                print_summary(mode, evaluation, args.top_k)
            except Exception as exc:
                errors[mode] = str(exc)
                print(f"{mode:<16}SKIP: {exc}", file=sys.stderr)
            finally:
                if reranker is not None:
                    reranker.close()
    finally:
        store.close()

    if not results:
        print("没有任何检索模式完成评测。", file=sys.stderr)
        return 1

    if args.show_misses:
        for evaluation in results.values():
            print_misses(evaluation)

    if args.output is not None:
        report = {
            "dataset": str(args.dataset),
            "case_count": len(cases),
            "top_k": args.top_k,
            "candidate_k": args.candidate_k,
            "results": {
                mode: evaluation.to_dict()
                for mode, evaluation in results.items()
            },
            "errors": errors,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"\n完整结果已写入: {args.output}")

    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
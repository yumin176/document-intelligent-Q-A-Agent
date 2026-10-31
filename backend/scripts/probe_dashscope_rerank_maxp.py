"""精排探针 V2：句子级 MaxP 重排。

固定窗口切出的块常把答案放在块尾、块首是与问题无关的上下文。整块送 rerank 时，
语义会被块首噪声拉低。这里把候选块拆成句子，逐句打分，取块内最高句分作为块分。
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import get_settings  # noqa: E402
from app.dependencies import get_retriever  # noqa: E402
from app.evaluation.retrieval import normalize_match_text  # noqa: E402

RERANK_URL = "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank"
SPLIT_RE = re.compile(r"[。！？；\n]+")


def split_sentences(text: str, max_sentences: int = 8) -> list[str]:
    parts = [p.strip() for p in SPLIT_RE.split(text)]
    # 丢掉过短的碎片（页眉页码、表格符号行）
    parts = [p for p in parts if len(normalize_match_text(p)) >= 6]
    return parts[:max_sentences]


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
    parser.add_argument("--max-sentences", type=int, default=8)
    parser.add_argument("--output")
    args = parser.parse_args()

    key = get_settings().embedding_api_key
    cases = [json.loads(l) for l in Path(args.dataset).read_text("utf-8").splitlines() if l.strip()]
    retriever = get_retriever()

    before = after = 0
    rr_before: list[int] = []
    rr_after: list[int] = []
    changes: list[dict] = []
    top_sentence: list[str] = []

    with httpx.Client(timeout=60, trust_env=True) as client:
        for case in cases:
            hits = retriever.search(case["question"], top_k=args.pool)
            texts = [h.text for h in hits]
            ok_before, rank_before = hit_at_k(case, texts, args.top_k)
            before += ok_before
            if rank_before:
                rr_before.append(rank_before)

            sentences: list[str] = []
            owner: list[int] = []
            for idx, text in enumerate(texts):
                for sentence in split_sentences(text, args.max_sentences):
                    sentences.append(sentence)
                    owner.append(idx)

            scores = call_rerank(client, key, args.model, case["question"], sentences)
            chunk_scores: dict[int, float] = {}
            best_sentence: dict[int, str] = {}
            for score, chunk_idx, sentence in zip(scores, owner, sentences):
                if score > chunk_scores.get(chunk_idx, -1.0):
                    chunk_scores[chunk_idx] = score
                    best_sentence[chunk_idx] = sentence

            order = sorted(chunk_scores, key=lambda i: chunk_scores[i], reverse=True)
            reranked = [texts[i] for i in order]
            ok_after, rank_after = hit_at_k(case, reranked, args.top_k)
            after += ok_after
            if rank_after:
                rr_after.append(rank_after)
                top_sentence.append(best_sentence[order[0]][:70])

            if ok_before != ok_after:
                changes.append(
                    {
                        "case_id": case["id"],
                        "question": case["question"],
                        "before": ok_before,
                        "after": ok_after,
                        "rank_before": rank_before,
                        "rank_after": rank_after,
                    }
                )

    total = len(cases)
    print(f"样本 {total} | pool={args.pool} top_k={args.top_k} model={args.model} maxp={args.max_sentences}")
    print(f"未重排 Recall@{args.top_k}: {before / total:.3f}  MRR: {sum(1 / r for r in rr_before) / total:.3f}")
    print(f"MaxP 后 Recall@{args.top_k}: {after / total:.3f}  MRR: {sum(1 / r for r in rr_after) / total:.3f}")
    print(f"p50 排名: {statistics.median(rr_after) if rr_after else '-'}")
    print(f"\n口径变化 {len(changes)} 条:")
    for item in changes:
        print(f"  {'捞回' if item['after'] else '丢失'} {item['case_id']:16s} rank {item['rank_before']} → {item['rank_after']}  {item['question'][:36]}")

    if args.output:
        Path(args.output).write_text(
            json.dumps(
                {
                    "dataset": str(args.dataset),
                    "model": args.model,
                    "mode": "maxp",
                    "pool": args.pool,
                    "top_k": args.top_k,
                    "case_count": total,
                    "before_recall": before / total,
                    "after_recall": after / total,
                    "before_mrr": sum(1 / r for r in rr_before) / total,
                    "after_mrr": sum(1 / r for r in rr_after) / total,
                    "changes": changes,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

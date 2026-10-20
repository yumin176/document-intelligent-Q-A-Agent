import json
import math

import pytest

from app.evaluation.retrieval import (
    RelevantChunk,
    RetrievalCase,
    evaluate_retriever,
    hit_at_k,
    load_cases,
    mrr_at_k,
    ndcg_at_k,
    recall_at_k,
)
from app.retrieval.base import Retriever
from app.vectorstore.base import SearchResult


def test_retrieval_metrics() -> None:
    ranked = [("js.md", 2), ("js.md", 0), ("js.md", 1)]
    relevant = [("js.md", 0), ("js.md", 1)]

    assert recall_at_k(ranked, relevant, 3) == 1.0
    assert hit_at_k(ranked, relevant, 1) == 0.0
    assert mrr_at_k(ranked, relevant, 3) == pytest.approx(0.5)
    assert ndcg_at_k(ranked, relevant, 3) == pytest.approx(
        (1 / math.log2(3) + 1 / 2) / (1 + 1 / math.log2(3))
    )


def test_retrieval_metrics_without_hit() -> None:
    ranked = [("js.md", 9)]
    relevant = [("js.md", 0)]

    assert recall_at_k(ranked, relevant, 5) == 0.0
    assert hit_at_k(ranked, relevant, 5) == 0.0
    assert mrr_at_k(ranked, relevant, 5) == 0.0
    assert ndcg_at_k(ranked, relevant, 5) == 0.0


def test_load_cases_from_jsonl(tmp_path) -> None:
    path = tmp_path / "cases.jsonl"
    path.write_text(
        json.dumps(
            {
                "id": "q1",
                "question": "问题",
                "relevant": [{"filename": "JS.MD", "chunk_index": 0}],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    cases = load_cases(path)

    assert cases == (
        RetrievalCase(
            case_id="q1",
            question="问题",
            relevant=(RelevantChunk(filename="js.md", chunk_index=0),),
        ),
    )


def test_load_cases_rejects_duplicate_ids(tmp_path) -> None:
    path = tmp_path / "cases.jsonl"
    line = json.dumps(
        {
            "id": "q1",
            "question": "问题",
            "relevant": [{"filename": "js.md", "chunk_index": 0}],
        },
        ensure_ascii=False,
    )
    path.write_text(f"{line}\n{line}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="重复 id"):
        load_cases(path)


class FakeRetriever(Retriever):
    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        return [
            SearchResult(
                id="wrong",
                score=0.9,
                text="无关内容",
                metadata={"filename": "js.md", "chunk_index": 9},
            ),
            SearchResult(
                id="right",
                score=0.8,
                text="相关内容",
                metadata={"filename": "js.md", "chunk_index": 0},
            ),
        ][:top_k]


def test_evaluate_retriever_records_rank_and_metrics() -> None:
    cases = (
        RetrievalCase(
            case_id="q1",
            question="测试问题",
            relevant=(RelevantChunk(filename="js.md", chunk_index=0),),
        ),
    )

    result = evaluate_retriever(FakeRetriever(), cases, top_k=5)
    query = result.queries[0]

    assert result.mode == "FakeRetriever"
    assert query.retrieved_ids == ("wrong", "right")
    assert query.first_relevant_rank == 2
    assert query.recall == 1.0
    assert query.mrr == pytest.approx(0.5)
    assert result.summary["latency_ms"] >= 0
import json
import math
from pathlib import Path

import pytest

from app.evaluation.retrieval import (
    RetrievalCase,
    evaluate_retriever,
    hit_at_k,
    load_cases,
    matched_snippet_indexes,
    mrr_at_k,
    ndcg_at_k,
    normalize_match_text,
    recall_at_k,
)
from app.retrieval.base import Retriever
from app.vectorstore.base import SearchResult


EVALUATION_DIR = Path(__file__).resolve().parents[1] / "evaluation"


def make_case(
    *,
    doc: str = "guide/essentials/computed.md",
    snippets: tuple[str, ...] = ("计算属性值会基于其响应式依赖被缓存",),
) -> RetrievalCase:
    return RetrievalCase(
        case_id="q1",
        question="测试问题",
        relevant_doc=doc,
        relevant_snippets=snippets,
    )


def test_load_real_content_anchored_dataset() -> None:
    cases = load_cases(EVALUATION_DIR / "retrieval_cases_content.jsonl")

    assert len(cases) == 30
    assert len({case.case_id for case in cases}) == 30
    assert all(case.relevant_doc for case in cases)
    assert all(case.relevant_snippets for case in cases)


def test_load_content_case_from_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "cases.jsonl"
    path.write_text(
        json.dumps(
            {
                "id": "q1",
                "question": "问题",
                "relevant_doc": "技术文档-Vue3中文/guide/essentials/computed.md",
                "relevant_snippets": ["计算属性值会基于其响应式依赖被缓存"],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    cases = load_cases(path)

    assert cases[0].case_id == "q1"
    assert cases[0].relevant_doc == "技术文档-vue3中文/guide/essentials/computed.md"
    assert cases[0].relevant_snippets == ("计算属性值会基于其响应式依赖被缓存",)


def test_load_cases_rejects_old_chunk_index_format(tmp_path: Path) -> None:
    path = tmp_path / "cases.jsonl"
    path.write_text(
        json.dumps(
            {
                "id": "q1",
                "question": "问题",
                "relevant": [{"filename": "js.md", "chunk_index": 0}],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="旧格式"):
        load_cases(path)


def test_load_cases_rejects_duplicate_ids(tmp_path: Path) -> None:
    path = tmp_path / "cases.jsonl"
    case = {
        "id": "q1",
        "question": "问题",
        "relevant_doc": "guide/computed.md",
        "relevant_snippets": ["计算属性值会基于其响应式依赖被缓存"],
    }
    line = json.dumps(case, ensure_ascii=False)
    path.write_text(f"{line}\n{line}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="重复 id"):
        load_cases(path)


def test_load_cases_rejects_missing_doc(tmp_path: Path) -> None:
    path = tmp_path / "cases.jsonl"
    path.write_text(
        json.dumps(
            {
                "id": "q1",
                "question": "问题",
                "relevant_doc": "",
                "relevant_snippets": ["内容"],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="relevant_doc"):
        load_cases(path)


def test_load_cases_rejects_empty_snippets(tmp_path: Path) -> None:
    path = tmp_path / "cases.jsonl"
    path.write_text(
        json.dumps(
            {
                "id": "q1",
                "question": "问题",
                "relevant_doc": "guide/computed.md",
                "relevant_snippets": [],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="relevant_snippets"):
        load_cases(path)


def test_load_cases_rejects_duplicate_snippets(tmp_path: Path) -> None:
    path = tmp_path / "cases.jsonl"
    path.write_text(
        json.dumps(
            {
                "id": "q1",
                "question": "问题",
                "relevant_doc": "guide/computed.md",
                "relevant_snippets": ["计算属性 缓存", "计算属性缓存"],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="重复片段"):
        load_cases(path)


def test_normalize_match_text_removes_whitespace() -> None:
    assert normalize_match_text("计算 属性\n值") == "计算属性值"


def test_matched_snippets_ignore_whitespace_and_path_case() -> None:
    case = make_case(
        snippets=(
            "计算属性值会基于其响应式依赖被缓存",
            "计算属性的 getter 应只做计算",
        )
    )
    hit = SearchResult(
        id="right",
        score=0.9,
        text="计算属性值会基于\n其响应式依赖被缓存；计算属性的 getter 应只做计算",
        metadata={"doc_name": "GUIDE\\essentials\\computed.md"},
    )

    assert matched_snippet_indexes(hit, case) == {0, 1}


def test_document_mismatch_does_not_match() -> None:
    case = make_case()
    hit = SearchResult(
        id="wrong-doc",
        score=0.9,
        text=case.relevant_snippets[0],
        metadata={"doc_name": "guide/essentials/other.md"},
    )

    assert matched_snippet_indexes(hit, case) == set()


def test_retrieval_metrics() -> None:
    ranked_matches = [set(), {0}, {1}]

    assert recall_at_k(ranked_matches, snippet_count=2, k=3) == 1.0
    assert hit_at_k(ranked_matches, k=1) == 0.0
    assert mrr_at_k(ranked_matches, k=3) == pytest.approx(0.5)
    assert ndcg_at_k(ranked_matches, snippet_count=2, k=3) == pytest.approx(
        (1 / math.log2(3) + 1 / 2) / (1 + 1 / math.log2(3))
    )


def test_retrieval_metrics_without_hit() -> None:
    ranked_matches = [set(), set()]

    assert recall_at_k(ranked_matches, snippet_count=2, k=5) == 0.0
    assert hit_at_k(ranked_matches, k=5) == 0.0
    assert mrr_at_k(ranked_matches, k=5) == 0.0
    assert ndcg_at_k(ranked_matches, snippet_count=2, k=5) == 0.0


class FakeRetriever(Retriever):
    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        return [
            SearchResult(
                id="wrong",
                score=0.9,
                text="无关内容",
                metadata={"doc_name": "guide/essentials/other.md"},
            ),
            SearchResult(
                id="right",
                score=0.8,
                text=(
                    "计算属性值会基于\n其响应式依赖被缓存；"
                    "计算属性的 getter 应只做计算"
                ),
                metadata={"doc_name": "guide/essentials/computed.md"},
            ),
        ][:top_k]


def test_evaluate_retriever_records_snippet_coverage() -> None:
    case = make_case(
        snippets=(
            "计算属性值会基于其响应式依赖被缓存",
            "计算属性的 getter 应只做计算",
        )
    )

    result = evaluate_retriever(FakeRetriever(), (case,), top_k=5)
    query = result.queries[0]

    assert result.mode == "FakeRetriever"
    assert query.retrieved_ids == ("wrong", "right")
    assert query.retrieved_docs == (
        "guide/essentials/other.md",
        "guide/essentials/computed.md",
    )
    assert query.matched_snippets == case.relevant_snippets
    assert query.first_relevant_rank == 2
    assert query.recall == 1.0
    assert query.hit == 1.0
    assert query.mrr == pytest.approx(0.5)
    assert query.ndcg == pytest.approx((1 / math.log2(3)) / (1 + 1 / math.log2(3)))
    assert result.summary["latency_ms"] >= 0


def test_evaluation_result_is_json_serializable() -> None:
    result = evaluate_retriever(FakeRetriever(), (make_case(),), top_k=5)

    payload = json.dumps(result.to_dict(), ensure_ascii=False)
    assert '"matched_snippets"' in payload
    assert '"retrieved_docs"' in payload
"""内容锚点检索数据集、指标与评测运行器。"""

from __future__ import annotations

import json
import math
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable, Sequence

from app.retrieval.base import Retriever
from app.vectorstore.base import SearchResult


@dataclass(frozen=True)
class RetrievalCase:
    """一条检索评测样本，使用内容片段作为相关性锚点。"""

    case_id: str
    question: str
    relevant_doc: str
    relevant_snippets: tuple[str, ...]


@dataclass(frozen=True)
class QueryEvaluation:
    """单条问题的评测结果。"""

    case_id: str
    question: str
    retrieved_ids: tuple[str, ...]
    retrieved_docs: tuple[str, ...]
    relevant_snippets: tuple[str, ...]
    matched_snippets: tuple[str, ...]
    first_relevant_rank: int | None
    recall: float
    hit: float
    mrr: float
    ndcg: float
    latency_ms: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "question": self.question,
            "retrieved_ids": list(self.retrieved_ids),
            "retrieved_docs": list(self.retrieved_docs),
            "relevant_snippets": list(self.relevant_snippets),
            "matched_snippets": list(self.matched_snippets),
            "first_relevant_rank": self.first_relevant_rank,
            "recall": self.recall,
            "hit": self.hit,
            "mrr": self.mrr,
            "ndcg": self.ndcg,
            "latency_ms": self.latency_ms,
        }


@dataclass(frozen=True)
class ModeEvaluation:
    """一种检索模式的完整评测结果。"""

    mode: str
    queries: tuple[QueryEvaluation, ...]
    elapsed_ms: float

    @property
    def summary(self) -> dict[str, float]:
        return {
            "recall": _mean(query.recall for query in self.queries),
            "hit": _mean(query.hit for query in self.queries),
            "mrr": _mean(query.mrr for query in self.queries),
            "ndcg": _mean(query.ndcg for query in self.queries),
            "latency_ms": _mean(query.latency_ms for query in self.queries),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "summary": self.summary,
            "elapsed_ms": self.elapsed_ms,
            "queries": [query.to_dict() for query in self.queries],
        }


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        return 0.0
    return sum(values) / len(values)


def normalize_doc_key(value: str) -> str:
    """统一语料库相对路径，兼容 Windows 和 POSIX 分隔符。"""
    normalized = unicodedata.normalize("NFKC", value).strip().replace("\\", "/")
    while normalized.startswith("./"):
        normalized = normalized[2:]
    return normalized.casefold()


def normalize_match_text(text: str) -> str:
    """匹配时忽略 Unicode 兼容差异和所有空白字符。"""
    normalized = unicodedata.normalize("NFKC", text)
    return "".join(normalized.split())


def result_doc_name(result: SearchResult) -> str | None:
    """优先使用语料库相对路径 doc_name，旧数据回退到 filename。"""
    metadata = result.metadata or {}
    doc_name = metadata.get("doc_name") or metadata.get("filename")
    if not isinstance(doc_name, str) or not doc_name.strip():
        return None
    return normalize_doc_key(doc_name)


def matched_snippet_indexes(
    result: SearchResult,
    case: RetrievalCase,
) -> set[int]:
    """返回该结果命中的相关 snippet 索引。"""
    if result_doc_name(result) != normalize_doc_key(case.relevant_doc):
        return set()

    normalized_text = normalize_match_text(result.text)
    return {
        index
        for index, snippet in enumerate(case.relevant_snippets)
        if normalize_match_text(snippet) in normalized_text
    }


def _parse_case(raw: Any, location: str) -> RetrievalCase:
    if not isinstance(raw, dict):
        raise ValueError(f"{location}: 样本必须是对象")
    if "relevant" in raw:
        raise ValueError(
            f"{location}: 旧格式 relevant/chunk_index 已不再支持，"
            "请使用 relevant_doc/relevant_snippets"
        )

    case_id = raw.get("id")
    if not isinstance(case_id, str) or not case_id.strip():
        raise ValueError(f"{location}: id 必须是非空字符串")

    question = raw.get("question")
    if not isinstance(question, str) or not question.strip():
        raise ValueError(f"{location}: question 必须是非空字符串")

    relevant_doc = raw.get("relevant_doc")
    if not isinstance(relevant_doc, str) or not relevant_doc.strip():
        raise ValueError(f"{location}: relevant_doc 必须是非空字符串")

    raw_snippets = raw.get("relevant_snippets")
    if not isinstance(raw_snippets, list) or not raw_snippets:
        raise ValueError(f"{location}: relevant_snippets 必须是非空数组")

    snippets: list[str] = []
    normalized_snippets: set[str] = set()
    for index, raw_snippet in enumerate(raw_snippets):
        if not isinstance(raw_snippet, str) or not raw_snippet.strip():
            raise ValueError(
                f"{location}.relevant_snippets[{index}]: snippet 必须是非空字符串"
            )
        snippet = raw_snippet.strip()
        normalized = normalize_match_text(snippet)
        if normalized in normalized_snippets:
            raise ValueError(f"{location}: relevant_snippets 中存在重复片段")
        normalized_snippets.add(normalized)
        snippets.append(snippet)

    return RetrievalCase(
        case_id=case_id.strip(),
        question=question.strip(),
        relevant_doc=normalize_doc_key(relevant_doc),
        relevant_snippets=tuple(snippets),
    )


def load_cases(path: Path) -> tuple[RetrievalCase, ...]:
    """从 JSONL 或 JSON 数组加载内容锚点评测集。"""
    try:
        text = path.read_text(encoding="utf-8-sig")
    except OSError as exc:
        raise ValueError(f"无法读取评测集: {path}") from exc

    if path.suffix.lower() == ".json":
        try:
            raw_cases = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"JSON 格式错误: {exc}") from exc
        if not isinstance(raw_cases, list):
            raise ValueError("JSON 评测集必须是数组")
        located_cases = [
            (raw_case, f"item {index}")
            for index, raw_case in enumerate(raw_cases)
        ]
    else:
        located_cases = []
        for line_number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                raw_case = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"第 {line_number} 行 JSON 格式错误: {exc}") from exc
            located_cases.append((raw_case, f"line {line_number}"))

    if not located_cases:
        raise ValueError("评测集不能为空")

    cases = tuple(
        _parse_case(raw_case, location)
        for raw_case, location in located_cases
    )
    case_ids = [case.case_id for case in cases]
    if len(set(case_ids)) != len(case_ids):
        raise ValueError("评测集中存在重复 id")
    return cases


def recall_at_k(
    ranked_matches: Sequence[set[int]],
    snippet_count: int,
    k: int,
) -> float:
    """top-k 覆盖的不同 snippet 占全部 snippet 的比例。"""
    if k <= 0 or snippet_count <= 0:
        return 0.0
    covered: set[int] = set()
    for matches in ranked_matches[:k]:
        covered.update(matches)
    return len(covered) / snippet_count


def hit_at_k(ranked_matches: Sequence[set[int]], k: int) -> float:
    if k <= 0:
        return 0.0
    return float(any(ranked_matches[:k]))


def mrr_at_k(ranked_matches: Sequence[set[int]], k: int) -> float:
    for rank, matches in enumerate(ranked_matches[:k], start=1):
        if matches:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(
    ranked_matches: Sequence[set[int]],
    snippet_count: int,
    k: int,
) -> float:
    """将包含任意相关 snippet 的结果视为二值相关。"""
    if k <= 0 or snippet_count <= 0:
        return 0.0
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, matches in enumerate(ranked_matches[:k], start=1)
        if matches
    )
    ideal_hits = min(snippet_count, k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg else 0.0


def evaluate_retriever(
    retriever: Retriever,
    cases: Sequence[RetrievalCase],
    top_k: int,
    mode: str | None = None,
) -> ModeEvaluation:
    """在同一批内容锚点问题上评测一个 Retriever。"""
    if top_k <= 0:
        raise ValueError("top_k 必须大于 0")
    if not cases:
        raise ValueError("评测集不能为空")

    started_at = perf_counter()
    evaluations: list[QueryEvaluation] = []

    for case in cases:
        query_started_at = perf_counter()
        hits = retriever.search(case.question, top_k=top_k)
        latency_ms = (perf_counter() - query_started_at) * 1000

        ranked_ids: list[str] = []
        ranked_docs: list[str] = []
        ranked_matches: list[set[int]] = []
        seen_ids: set[str] = set()

        for hit in hits:
            if hit.id in seen_ids:
                continue
            seen_ids.add(hit.id)
            ranked_ids.append(hit.id)
            ranked_docs.append(result_doc_name(hit) or "")
            ranked_matches.append(matched_snippet_indexes(hit, case))

        covered = set().union(*ranked_matches) if ranked_matches else set()
        first_relevant_rank = next(
            (
                rank
                for rank, matches in enumerate(ranked_matches[:top_k], start=1)
                if matches
            ),
            None,
        )
        matched_snippets = tuple(
            snippet
            for index, snippet in enumerate(case.relevant_snippets)
            if index in covered
        )

        evaluations.append(
            QueryEvaluation(
                case_id=case.case_id,
                question=case.question,
                retrieved_ids=tuple(ranked_ids),
                retrieved_docs=tuple(ranked_docs),
                relevant_snippets=case.relevant_snippets,
                matched_snippets=matched_snippets,
                first_relevant_rank=first_relevant_rank,
                recall=recall_at_k(
                    ranked_matches,
                    len(case.relevant_snippets),
                    top_k,
                ),
                hit=hit_at_k(ranked_matches, top_k),
                mrr=mrr_at_k(ranked_matches, top_k),
                ndcg=ndcg_at_k(
                    ranked_matches,
                    len(case.relevant_snippets),
                    top_k,
                ),
                latency_ms=latency_ms,
            )
        )

    return ModeEvaluation(
        mode=mode or type(retriever).__name__,
        queries=tuple(evaluations),
        elapsed_ms=(perf_counter() - started_at) * 1000,
    )
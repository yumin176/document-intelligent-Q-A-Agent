"""检索数据集、指标与评测运行器。"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable, Sequence

from app.ingestion.service import normalize_filename
from app.retrieval.base import Retriever
from app.vectorstore.base import SearchResult

ChunkKey = tuple[str, int]


@dataclass(frozen=True)
class RelevantChunk:
    """评测集中标注为相关的 chunk。"""

    filename: str
    chunk_index: int

    @property
    def key(self) -> ChunkKey:
        return self.filename, self.chunk_index


@dataclass(frozen=True)
class RetrievalCase:
    """一条检索评测样本。"""

    case_id: str
    question: str
    relevant: tuple[RelevantChunk, ...]


@dataclass(frozen=True)
class QueryEvaluation:
    """单条问题的评测结果。"""

    case_id: str
    question: str
    retrieved_ids: tuple[str, ...]
    retrieved_keys: tuple[ChunkKey, ...]
    relevant_keys: tuple[ChunkKey, ...]
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
            "retrieved_keys": [list(key) for key in self.retrieved_keys],
            "relevant_keys": [list(key) for key in self.relevant_keys],
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


def _parse_relevant_chunk(raw: Any, location: str) -> RelevantChunk:
    if not isinstance(raw, dict):
        raise ValueError(f"{location}: relevant 项必须是对象")

    filename = raw.get("filename")
    if not isinstance(filename, str) or not filename.strip():
        raise ValueError(f"{location}: filename 必须是非空字符串")

    raw_index = raw.get("chunk_index")
    if isinstance(raw_index, bool):
        raise ValueError(f"{location}: chunk_index 必须是整数")
    try:
        chunk_index = int(raw_index)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{location}: chunk_index 必须是整数") from exc
    if chunk_index < 0:
        raise ValueError(f"{location}: chunk_index 不能小于 0")

    return RelevantChunk(
        filename=normalize_filename(filename),
        chunk_index=chunk_index,
    )


def _parse_case(raw: Any, location: str) -> RetrievalCase:
    if not isinstance(raw, dict):
        raise ValueError(f"{location}: 样本必须是对象")

    case_id = raw.get("id")
    if not isinstance(case_id, str) or not case_id.strip():
        raise ValueError(f"{location}: id 必须是非空字符串")

    question = raw.get("question")
    if not isinstance(question, str) or not question.strip():
        raise ValueError(f"{location}: question 必须是非空字符串")

    raw_relevant = raw.get("relevant")
    if not isinstance(raw_relevant, list) or not raw_relevant:
        raise ValueError(f"{location}: relevant 必须是非空数组")

    relevant = tuple(
        _parse_relevant_chunk(item, f"{location}.relevant[{index}]")
        for index, item in enumerate(raw_relevant)
    )
    if len({item.key for item in relevant}) != len(relevant):
        raise ValueError(f"{location}: relevant 中存在重复 chunk")

    return RetrievalCase(
        case_id=case_id.strip(),
        question=question.strip(),
        relevant=relevant,
    )


def load_cases(path: Path) -> tuple[RetrievalCase, ...]:
    """从 JSONL 或 JSON 数组加载评测集。"""
    try:
        text = path.read_text(encoding="utf-8-sig")
    except OSError as exc:
        raise ValueError(f"无法读取评测集: {path}") from exc

    if path.suffix.lower() == ".json":
        try:
            raw_cases = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"评测集 JSON 格式错误: {exc}") from exc
        if not isinstance(raw_cases, list):
            raise ValueError("评测集 JSON 顶层必须是数组")
        located_cases = [
            (raw_case, f"case[{index}]")
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


def result_key(result: SearchResult) -> ChunkKey | None:
    """从检索结果中提取稳定的文件名和 chunk 序号。"""
    metadata = result.metadata or {}
    filename = metadata.get("filename")
    raw_index = metadata.get("chunk_index")

    if not isinstance(filename, str) or not filename.strip():
        return None
    if isinstance(raw_index, bool):
        return None
    try:
        chunk_index = int(raw_index)
    except (TypeError, ValueError):
        return None
    if chunk_index < 0:
        return None

    return normalize_filename(filename), chunk_index


def recall_at_k(
    ranked: Sequence[ChunkKey],
    relevant: Sequence[ChunkKey],
    k: int,
) -> float:
    """top-k 中命中的相关 chunk 占全部相关 chunk 的比例。"""
    if k <= 0 or not relevant:
        return 0.0
    relevant_set = set(relevant)
    return len(set(ranked[:k]) & relevant_set) / len(relevant_set)


def hit_at_k(
    ranked: Sequence[ChunkKey],
    relevant: Sequence[ChunkKey],
    k: int,
) -> float:
    """top-k 中是否至少命中一个相关 chunk。"""
    if k <= 0 or not relevant:
        return 0.0
    return float(bool(set(ranked[:k]) & set(relevant)))


def mrr_at_k(
    ranked: Sequence[ChunkKey],
    relevant: Sequence[ChunkKey],
    k: int,
) -> float:
    """第一个相关结果的倒数排名。"""
    relevant_set = set(relevant)
    for rank, key in enumerate(ranked[:k], start=1):
        if key in relevant_set:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(
    ranked: Sequence[ChunkKey],
    relevant: Sequence[ChunkKey],
    k: int,
) -> float:
    """二值相关性下的 nDCG@k。"""
    if k <= 0 or not relevant:
        return 0.0

    relevant_set = set(relevant)
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, key in enumerate(ranked[:k], start=1)
        if key in relevant_set
    )
    ideal_hits = min(len(relevant_set), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg else 0.0


def evaluate_retriever(
    retriever: Retriever,
    cases: Sequence[RetrievalCase],
    top_k: int,
    mode: str | None = None,
) -> ModeEvaluation:
    """在同一批问题上评测一个 Retriever。"""
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

        seen_keys: set[ChunkKey] = set()
        ranked_keys: list[ChunkKey] = []
        for hit in hits:
            key = result_key(hit)
            if key is None or key in seen_keys:
                continue
            seen_keys.add(key)
            ranked_keys.append(key)

        relevant_keys = [item.key for item in case.relevant]
        relevant_set = set(relevant_keys)
        first_relevant_rank = next(
            (
                rank
                for rank, key in enumerate(ranked_keys[:top_k], start=1)
                if key in relevant_set
            ),
            None,
        )

        evaluations.append(
            QueryEvaluation(
                case_id=case.case_id,
                question=case.question,
                retrieved_ids=tuple(hit.id for hit in hits),
                retrieved_keys=tuple(ranked_keys),
                relevant_keys=tuple(relevant_keys),
                first_relevant_rank=first_relevant_rank,
                recall=recall_at_k(ranked_keys, relevant_keys, top_k),
                hit=hit_at_k(ranked_keys, relevant_keys, top_k),
                mrr=mrr_at_k(ranked_keys, relevant_keys, top_k),
                ndcg=ndcg_at_k(ranked_keys, relevant_keys, top_k),
                latency_ms=latency_ms,
            )
        )

    return ModeEvaluation(
        mode=mode or type(retriever).__name__,
        queries=tuple(evaluations),
        elapsed_ms=(perf_counter() - started_at) * 1000,
    )
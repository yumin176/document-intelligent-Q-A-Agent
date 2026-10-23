"""Grader：判断检索候选是否与问题相关。"""

import json
import logging

from app.llm.base import ChatModel
from app.prompts.templates import build_grader_prompt
from app.vectorstore.base import SearchResult

logger = logging.getLogger(__name__)


def parse_relevant_indices(raw: str, candidate_count: int) -> list[int]:
    """解析 Grader 返回的 1-based 候选编号。

    返回去重后的编号列表；最终输出顺序由 grade_chunks 保证。
    """
    if candidate_count <= 0:
        raise ValueError("候选数量必须大于 0")

    try:
        start_index = raw.find("{")
        end_index = raw.rfind("}")
        if start_index == -1 or end_index <= start_index:
            raise ValueError("输出中没有 JSON 对象")

        candidate_msg = json.loads(raw[start_index : end_index + 1])
        if not isinstance(candidate_msg, dict):
            raise ValueError("Grader 输出必须是对象")

        relevant_indices = candidate_msg.get("relevant_indices")
        if not isinstance(relevant_indices, list):
            raise ValueError("relevant_indices 必须是数组")

        unique_indices: set[int] = set()
        for index in relevant_indices:
            if isinstance(index, bool) or not isinstance(index, int):
                raise ValueError("候选编号必须是整数")
            if not 1 <= index <= candidate_count:
                raise ValueError("候选编号越界")
            unique_indices.add(index)

        return sorted(unique_indices)
    except (ValueError, TypeError) as exc:
        logger.warning("Grader 输出无效: %s", exc)
        raise ValueError("无法解析 relevant_indices") from exc


def grade_chunks(
    question: str,
    candidates: list[SearchResult],
    chat_model: ChatModel,
) -> list[SearchResult]:
    """调用 LLM 过滤候选 chunk，异常时保留全部候选。"""
    if not question.strip() or not candidates:
        return []

    try:
        prompt = build_grader_prompt(question=question, candidates=candidates)
        response = chat_model.complete(prompt=prompt, temperature=0)
        index_list = parse_relevant_indices(
            raw=response,
            candidate_count=len(candidates),
        )

        selected_indices = set(index_list)
        return [
            candidate
            for index, candidate in enumerate(candidates, start=1)
            if index in selected_indices
        ]
    except Exception as exc:
        logger.warning("Grader 调用或解析失败，保留全部候选: %s", exc)
        return list(candidates)
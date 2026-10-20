import json
import logging

from app.llm.base import ChatMessage, ChatModel
from app.prompts.templates import build_rerank_prompt
from app.rerank.base import Reranker
from app.vectorstore.base import SearchResult

logger = logging.getLogger(__name__)


class LLMReranker(Reranker):
    """使用 LLM 对候选 chunk 做 listwise 重排。"""

    def __init__(self, chat_model: ChatModel, max_candidates: int = 10) -> None:
        if max_candidates <= 0:
            raise ValueError("最大候选数不能小于1")
        self.chat_model = chat_model
        self.max_candidates = max_candidates

    def rerank(
        self,
        query: str,
        candidates: list[SearchResult],
        top_k: int = 5,
    ) -> list[SearchResult]:
        if top_k <= 0 or not candidates:
            return []

        limited_candidates = candidates[: self.max_candidates]
        system_prompt = build_rerank_prompt(query=query, candidates=limited_candidates)

        raw = self.chat_model.chat(
            messages=[ChatMessage(role="user", content=system_prompt)],
            temperature=0,
        )

        try:
            scored_candidates = self._parse_and_validate(raw, limited_candidates)
        except Exception as exc:
            # 局部丢弃会破坏排序完整性，因此整批失败时回退到原始顺序。
            logger.warning("Rerank 输出无效，回退原始顺序: %s", exc)
            return candidates[:top_k]

        return scored_candidates[:top_k]

    def _parse_and_validate(
        self,
        raw: str,
        candidates: list[SearchResult],
    ) -> list[SearchResult]:
        """解析并严格校验 LLM 的结构化输出。"""
        text = raw.strip()

        # 兼容 ```json ... ``` 或前后夹杂说明文本的情况。
        start = text.find("[")
        end = text.rfind("]")
        if start == -1 or end == -1:
            raise ValueError("返回结果中找不到 JSON 数组")

        data = json.loads(text[start : end + 1])
        if not isinstance(data, list):
            raise ValueError("返回结果不是 JSON 数组")
        if len(data) != len(candidates):
            raise ValueError("输入输出数量不一致")

        candidate_by_id = {candidate.id: candidate for candidate in candidates}
        seen_ids: set[str] = set()
        results: list[SearchResult] = []

        for item in data:
            if not isinstance(item, dict):
                raise ValueError("数组项不是对象")

            target_id = item.get("id")
            if target_id not in candidate_by_id:
                raise ValueError("存在无效 id")
            if target_id in seen_ids:
                raise ValueError("存在重复 id")

            try:
                score = float(item.get("score"))
            except (TypeError, ValueError) as exc:
                raise ValueError("score 不是数字") from exc

            if not 0 <= score <= 10:
                raise ValueError("score 超出 0-10 范围")

            candidate = candidate_by_id[target_id]
            results.append(
                SearchResult(
                    id=candidate.id,
                    score=score,
                    text=candidate.text,
                    metadata=candidate.metadata,
                )
            )
            seen_ids.add(target_id)

        if seen_ids != set(candidate_by_id):
            raise ValueError("存在未评分的候选")

        results.sort(key=lambda item: item.score, reverse=True)
        return results
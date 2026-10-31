"""阿里云百炼（DashScope）文本重排。

协议要点：
- 请求体是 ``{"model", "input": {"query", "documents"}, "parameters": {"top_n"}}``
- 响应分数在 ``output.results[].relevance_score``

失败一律回退原始顺序，但会记录 ``fallback_count``，避免把
"未重排的结果"当成精排结果统计。
"""

from __future__ import annotations

import logging

import httpx

from app.config import get_settings
from app.rerank.base import Reranker
from app.vectorstore.base import SearchResult

settings = get_settings()
logger = logging.getLogger(__name__)


class DashScopeReranker(Reranker):
    """通过百炼 text-rerank 接口对候选 chunk 重排。"""

    def __init__(
        self,
        api_url: str | None = settings.rerank_api_url,
        api_key: str | None = settings.rerank_api_key,
        model: str | None = settings.rerank_model,
        timeout: float | None = settings.rerank_timeout,
        max_candidates: int = 30,
        client: httpx.Client | None = None,
    ) -> None:
        if not (api_url and api_key and model):
            raise ValueError("rerank 配置不完整")
        if max_candidates <= 0:
            raise ValueError("最大候选数不能小于1")

        self.api_url = api_url
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.max_candidates = max_candidates
        self.client = client or httpx.Client(timeout=timeout)
        self._owns_client = client is None
        self.fallback_count = 0
        self.last_fallback_error: str | None = None

    def rerank(
        self,
        query: str,
        candidates: list[SearchResult],
        top_k: int = 5,
    ) -> list[SearchResult]:
        if top_k <= 0 or not candidates:
            return []

        limited_candidates = candidates[: self.max_candidates]
        expected_count = min(top_k, len(limited_candidates))

        payload = {
            "model": self.model,
            "input": {
                "query": query,
                "documents": [candidate.text for candidate in limited_candidates],
            },
            "parameters": {
                "return_documents": False,
                "top_n": expected_count,
            },
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        try:
            response = self.client.post(self.api_url, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
            scored_candidates = self._parse_and_validate(
                data["output"]["results"],
                limited_candidates,
                expected_count,
            )
        except Exception as exc:
            self.fallback_count += 1
            self.last_fallback_error = f"{type(exc).__name__}: {exc}"
            logger.warning("DashScope Rerank 请求或输出无效，回退原始顺序: %s", exc)
            return candidates[:top_k]

        return scored_candidates[:top_k]

    def close(self) -> None:
        if self._owns_client:
            self.client.close()

    def _parse_and_validate(
        self,
        response_results: list,
        candidates: list[SearchResult],
        expected_count: int,
    ) -> list[SearchResult]:
        if not isinstance(response_results, list) or not response_results:
            raise ValueError("返回结果不是非空数组")
        if len(response_results) != expected_count:
            raise ValueError("返回结果数量不符合预期")

        seen_indexes: set[int] = set()
        results: list[SearchResult] = []

        for item in response_results:
            if not isinstance(item, dict):
                raise ValueError("数组项不是对象")

            target_index = item.get("index")
            if (
                not isinstance(target_index, int)
                or isinstance(target_index, bool)
                or not 0 <= target_index < len(candidates)
            ):
                raise ValueError("存在无效 index")
            if target_index in seen_indexes:
                raise ValueError("存在重复 index")

            try:
                score = float(item.get("relevance_score"))
            except (TypeError, ValueError) as exc:
                raise ValueError("score 不是数字") from exc

            candidate = candidates[target_index]
            results.append(
                SearchResult(
                    id=candidate.id,
                    score=score,
                    text=candidate.text,
                    metadata=candidate.metadata,
                )
            )
            seen_indexes.add(target_index)

        results.sort(key=lambda item: item.score, reverse=True)
        return results

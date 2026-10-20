import logging

import httpx

from app.config import get_settings
from app.rerank.base import Reranker
from app.vectorstore.base import SearchResult

settings = get_settings()
logger = logging.getLogger(__name__)


class CohereReranker(Reranker):
    """通过 Cohere 兼容的 Rerank API 对候选 chunk 进行重排。"""

    def __init__(
        self,
        api_url: str | None = settings.rerank_api_url,
        api_key: str | None = settings.rerank_api_key,
        model: str | None = settings.rerank_model,
        timeout: float | None = settings.rerank_timeout,
        max_candidates: int = 10,
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
            "query": query,
            "documents": [candidate.text for candidate in limited_candidates],
            "top_n": expected_count,
            "return_documents": False,
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        try:
            response = self.client.post(
                self.api_url,
                headers=headers,
                json=payload,
            )
            response.raise_for_status()
            # Response对象的json方法，用于将响应内容从json格式转换为python字典
            data = response.json()
            scored_candidates = self._parse_and_validate(
                data["results"],
                limited_candidates,
                expected_count,
            )
        except Exception as exc:
            logger.warning("Rerank 请求或输出无效，回退原始顺序: %s", exc)
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

            if not 0 <= score <= 1:
                raise ValueError("score 超出 0-1 范围")

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

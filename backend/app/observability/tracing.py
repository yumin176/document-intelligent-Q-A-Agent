"""LangSmith 全链路追踪的薄封装。

设计约束：
- 默认完全关闭：未配置 API key 或 LANGSMITH_TRACING=false 时，本模块所有
  函数都是透传，不创建任何网络连接（保证测试与离线环境安全）。
- 分层：一条 trace 对应一次问答请求，chat.py 用 trace_request() 开根 span；
  Agent 图的 router / retrieve / grade / rewrite 由 LangGraph 原生上报
  （启用追踪后自动成为子 span），不再手动重复埋点；LLM 生成用
  trace_generation() 挂 llm span（我们是自研 ChatModel，原生不会上报）。
- 载荷脱敏：原生节点 span 会把整个 AgentState 当输出（含 50 条候选正文），
  直接用会让单条 trace 膨胀到几百 KB，因此 Client 上挂 hide_inputs /
  hide_outputs，把候选 chunk 压成 {id, score}。
- 上报走后台批量线程（Client 默认 auto_batch），不阻塞问答主链路；
  上报失败只记 debug 日志，绝不影响回答本身。
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from functools import lru_cache
from typing import Any

from langsmith import Client, trace, tracing_context
from langsmith.run_helpers import get_current_run_tree

from app.config import get_settings

logger = logging.getLogger(__name__)


def _current_run_tree() -> Any:
    """当前上下文里的 run tree（没有外层请求时为 None）。"""
    try:
        return get_current_run_tree()
    except Exception:  # pragma: no cover - 版本差异兜底
        return None

# 这些键里装的是候选 chunk 列表，整段正文对排障没用，只会撑大 trace
_HEAVY_CHUNK_KEYS = {"candidate_chunks", "relevant_chunks"}


def _chunk_brief(item: Any) -> Any:
    """把一条 SearchResult（对象或 dict）压成 {id, score}。"""
    if isinstance(item, dict):
        return {"id": item.get("id"), "score": item.get("score")}
    return {"id": getattr(item, "id", None), "score": getattr(item, "score", None)}


def _scrub_run_data(data: Any) -> Any:
    """去掉 span 载荷里的 chunk 正文，只保留定位与分数。

    注意：LangGraph 节点返回的是 SearchResult 对象而非 dict，
    因此这里不能只判断 dict，否则会把候选整段丢掉（看起来像检索为空）。
    """
    if isinstance(data, dict):
        cleaned: dict[str, Any] = {}
        for key, value in data.items():
            if key == "text" and isinstance(value, str):
                continue
            if key in _HEAVY_CHUNK_KEYS and isinstance(value, list):
                cleaned[key] = [_chunk_brief(item) for item in value]
                continue
            cleaned[key] = _scrub_run_data(value)
        return cleaned
    if isinstance(data, list):
        return [_scrub_run_data(item) for item in data]
    return data


@lru_cache(maxsize=1)
def _get_client() -> Client | None:
    """返回 LangSmith 客户端；未启用追踪时返回 None（调用方按关闭处理）。"""
    settings = get_settings()
    if not settings.langsmith_tracing or not settings.langsmith_api_key:
        return None
    try:
        return Client(
            api_url=settings.langsmith_endpoint,
            api_key=settings.langsmith_api_key,
            timeout_ms=5000,
            hide_inputs=_scrub_run_data,
            hide_outputs=_scrub_run_data,
        )
    except Exception:  # pragma: no cover - 构造失败时降级为关闭
        logger.warning("LangSmith 客户端初始化失败，追踪已禁用", exc_info=True)
        return None


def is_enabled() -> bool:
    """当前是否启用了 LangSmith 追踪。"""
    return _get_client() is not None


def reset_client_cache() -> None:
    """清空客户端缓存，供测试在修改配置后调用。"""
    _get_client.cache_clear()


@contextmanager
def trace_request(
    question: str,
    top_k: int,
    request_id: str,
) -> Iterator[Any]:
    """根 span：一次问答请求对应 LangSmith 上的一条 trace。

    未启用追踪时 yield None；启用时 yield 根 RunTree，
    调用方可用 record_output() 往根 span 补充输出字段。
    """
    client = _get_client()
    if client is None:
        yield None
        return

    settings = get_settings()
    metadata = {
        "request_id": request_id,
        "retrieval_mode": settings.retrieval_mode,
        "retrieval_candidate_k": settings.retrieval_candidate_k,
        "milvus_collection": settings.milvus_collection_name,
        "agent_max_rewrites": settings.agent_max_rewrites,
        "app_version": settings.app_version,
    }
    with tracing_context(
        enabled=True, client=client, project_name=settings.langsmith_project
    ):
        with trace(
            name="AgentService.answer",
            run_type="chain",
            inputs={"question": question, "top_k": top_k},
            metadata=metadata,
            client=client,
        ) as run:
            yield run


@contextmanager
def trace_generation(
    messages: list[dict[str, str]],
    stream: bool = False,
) -> Iterator[Any]:
    """LLM 生成的子 span（run_type=llm）；未启用追踪时 yield None。

    注意：这里**不再新开 tracing_context**。嵌套的 tracing_context 会把
    父 run tree 重置为 None，导致 generate 变成一条独立 trace。
    只有在没有外层请求上下文时（单独调用），才自己开一个。
    """
    client = _get_client()
    if client is None:
        yield None
        return

    settings = get_settings()
    ambient = _current_run_tree() is not None
    context = (
        nullcontext()
        if ambient
        else tracing_context(
            enabled=True, client=client, project_name=settings.langsmith_project
        )
    )
    with context:
        with trace(
            name="generate",
            run_type="llm",
            inputs={"messages": messages, "stream": stream},
            metadata={"llm_model": settings.llm_model},
            client=client,
        ) as run:
            yield run


def record_output(run: Any, **fields: Any) -> None:
    """向已开启的 span 补充输出字段；run 为 None（未启用追踪）时是 no-op。"""
    if run is None:
        return
    try:
        outputs = dict(run.outputs or {})
        outputs.update({k: v for k, v in fields.items() if v is not None})
        run.outputs = outputs
    except Exception:
        logger.debug("写入 LangSmith 输出失败", exc_info=True)

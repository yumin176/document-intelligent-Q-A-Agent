"""基于 LangGraph 的检索决策工作流。"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from langgraph.graph import END, START, StateGraph

from app.agent.state import AgentPlan, AgentState, Route
from app.llm.base import ChatModel
from app.prompts.templates import (
    build_grader_prompt,
    build_rewrite_prompt,
    build_router_prompt,
)
from app.retrieval.base import Retriever
from app.vectorstore.base import SearchResult

logger = logging.getLogger(__name__)


def extract_json(raw: str) -> Any:
    """从模型输出中提取 JSON，兼容 Markdown 代码块和前后说明文字。"""
    text = raw.strip()
    if not text:
        raise ValueError("模型输出为空")

    for opener, closer in (("{", "}"), ("[", "]")):
        start = text.find(opener)
        end = text.rfind(closer)
        if start == -1 or end <= start:
            continue
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            continue

    raise ValueError("模型输出中没有有效 JSON")


def parse_route(raw: str) -> Route:
    """解析 Router 输出的 route 字段。"""
    try:
        data = extract_json(raw)
    except ValueError:
        data = None

    if isinstance(data, dict) and data.get("route") in {"retrieve", "direct"}:
        return data["route"]

    match = re.search(r'"route"\s*:\s*"(retrieve|direct)"', raw, re.IGNORECASE)
    if match:
        return match.group(1).lower()  # type: ignore[return-value]

    raise ValueError("Router 输出中没有有效的 route")

def parse_relevant_indices(raw: str, candidate_count: int) -> list[int]:
    """解析 Grader 输出的 1-based 候选编号。"""
    if candidate_count <= 0:
        raise ValueError("候选数量必须大于 0")

    data = extract_json(raw)
    if isinstance(data, dict):
        raw_indices = data.get("relevant_indices")
    elif isinstance(data, list):
        raw_indices = data
    else:
        raise ValueError("Grader 输出必须是对象或数组")

    if not isinstance(raw_indices, list):
        raise ValueError("relevant_indices 必须是数组")

    indices: list[int] = []
    for value in raw_indices:
        if isinstance(value, bool):
            raise ValueError("候选编号不能是布尔值")
        try:
            index = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("候选编号必须是整数") from exc
        if not 1 <= index <= candidate_count:
            raise ValueError(f"候选编号越界: {index}")
        if index not in indices:
            indices.append(index)
    return indices


def parse_rewrite_query(raw: str) -> str:
    """解析 Rewrite 输出；同时兼容纯文本回答。"""
    try:
        data = extract_json(raw)
    except ValueError:
        data = None

    if isinstance(data, dict) and isinstance(data.get("query"), str):
        query = data["query"].strip()
    else:
        query = raw.strip()
        if query.startswith("```"):
            query = re.sub(r"^```(?:json|text)?\s*", "", query)
            query = re.sub(r"\s*```$", "", query).strip()
        query = query.strip().strip('"').strip("'").strip()

    if not query:
        raise ValueError("Rewrite 输出为空")
    return query[:300]


def heuristic_route(question: str) -> Route:
    """LLM 调用失败时使用的保守路由。"""
    normalized = re.sub(r"\s+", "", question).lower()
    direct_markers = {
        "你好",
        "您好",
        "hi",
        "hello",
        "谢谢",
        "多谢",
        "你是谁",
        "你能做什么",
        "介绍一下你自己",
    }
    if normalized in direct_markers:
        return "direct"
    return "retrieve"

class AgentRetrievalGraph:
    """Router → Retrieve → Grader → Rewrite 的检索决策图。"""

    def __init__(
        self,
        retriever: Retriever,
        chat_model: ChatModel,
        max_rewrites: int = 1,
        top_k: int = 4,
    ) -> None:
        if max_rewrites < 0:
            raise ValueError("max_rewrites 不能小于 0")
        if top_k <= 0:
            raise ValueError("top_k 必须大于 0")

        self.retriever = retriever
        self.chat_model = chat_model
        self.max_rewrites = max_rewrites
        self.top_k = top_k
        self.graph = self._build_graph()

    def plan(self, question: str, top_k: int | None = None) -> AgentPlan:
        """运行检索决策图，返回最终相关文档和执行轨迹。"""
        normalized_question = question.strip()
        if not normalized_question:
            raise ValueError("问题不能为空")

        k = self.top_k if top_k is None else top_k
        if k <= 0:
            raise ValueError("top_k 必须大于 0")

        state = self.graph.invoke(
            {
                "question": normalized_question,
                "search_query": normalized_question,
                "route": "retrieve",
                "hits": [],
                "relevant_hits": [],
                "retrieval_attempts": 0,
                "rewrite_count": 0,
                "max_rewrites": self.max_rewrites,
                "top_k": k,
                "query_history": [],
            },
            config={"recursion_limit": 10},
        )

        route = state.get("route", "retrieve")
        if route not in {"retrieve", "direct"}:
            route = "retrieve"

        return AgentPlan(
            question=normalized_question,
            route=route,
            search_query=state.get("search_query", normalized_question),
            hits=tuple(state.get("relevant_hits", []))
            if route == "retrieve"
            else (),
            retrieval_attempts=state.get("retrieval_attempts", 0),
            rewrite_count=state.get("rewrite_count", 0),
            query_history=tuple(state.get("query_history", [])),
        )

    def _build_graph(self):
        graph = StateGraph(AgentState)
        graph.add_node("router", self._router_node)
        graph.add_node("retrieve", self._retrieve_node)
        graph.add_node("grade", self._grade_node)
        graph.add_node("rewrite", self._rewrite_node)

        graph.add_edge(START, "router")
        graph.add_conditional_edges(
            "router",
            self._route_after_router,
            {"retrieve": "retrieve", "direct": END},
        )
        graph.add_edge("retrieve", "grade")
        graph.add_conditional_edges(
            "grade",
            self._route_after_grade,
            {"ready": END, "rewrite": "rewrite", "give_up": END},
        )
        graph.add_edge("rewrite", "retrieve")
        return graph.compile()
    def _router_node(self, state: AgentState) -> dict:
        question = state["question"]
        try:
            raw = self.chat_model.complete(
                build_router_prompt(question),
                temperature=0,
            )
            route = parse_route(raw)
        except Exception as exc:
            logger.warning("Router 输出无效，使用保守路由: %s", exc)
            route = heuristic_route(question)
        return {"route": route}

    def _retrieve_node(self, state: AgentState) -> dict:
        query = (state.get("search_query") or state["question"]).strip()
        hits = self.retriever.search(
            query,
            top_k=state.get("top_k", self.top_k),
        )

        unique_hits: list[SearchResult] = []
        seen_ids: set[str] = set()
        for hit in hits:
            if hit.id in seen_ids:
                continue
            seen_ids.add(hit.id)
            unique_hits.append(hit)

        query_history = [*state.get("query_history", []), query]
        return {
            "hits": unique_hits,
            "retrieval_attempts": state.get("retrieval_attempts", 0) + 1,
            "query_history": query_history,
        }

    def _grade_node(self, state: AgentState) -> dict:
        hits = state.get("hits", [])
        if not hits:
            return {"relevant_hits": []}

        try:
            raw = self.chat_model.complete(
                build_grader_prompt(state["question"], hits),
                temperature=0,
            )
            indices = parse_relevant_indices(raw, len(hits))
            relevant_hits = [hits[index - 1] for index in indices]
        except Exception as exc:
            logger.warning("Grader 输出无效，保留全部候选: %s", exc)
            relevant_hits = list(hits)

        return {"relevant_hits": relevant_hits}

    def _rewrite_node(self, state: AgentState) -> dict:
        question = state["question"]
        current_query = state.get("search_query") or question

        try:
            raw = self.chat_model.complete(
                build_rewrite_prompt(
                    question=question,
                    current_query=current_query,
                    candidates=state.get("hits", []),
                ),
                temperature=0,
            )
            rewritten_query = parse_rewrite_query(raw)
        except Exception as exc:
            logger.warning("Rewrite 输出无效，使用保守改写: %s", exc)
            rewritten_query = (
                question if current_query != question else f"{question} 相关概念"
            )

        if rewritten_query == current_query:
            rewritten_query = f"{question} 相关概念"

        return {
            "search_query": rewritten_query,
            "rewrite_count": state.get("rewrite_count", 0) + 1,
        }

    @staticmethod
    def _route_after_router(state: AgentState) -> str:
        return state.get("route", "retrieve")

    @staticmethod
    def _route_after_grade(state: AgentState) -> str:
        if state.get("relevant_hits"):
            return "ready"
        if state.get("retrieval_attempts", 0) <= state.get("max_rewrites", 0):
            return "rewrite"
        return "give_up"
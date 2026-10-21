"""Agent 工作流的状态与输出模型。"""

from dataclasses import dataclass
from typing import Literal, TypedDict

from app.vectorstore.base import SearchResult

Route = Literal["retrieve", "direct"]


class AgentState(TypedDict, total=False):
    """LangGraph 在节点之间传递的状态。"""

    question: str
    search_query: str
    route: Route
    hits: list[SearchResult]
    relevant_hits: list[SearchResult]
    retrieval_attempts: int
    rewrite_count: int
    max_rewrites: int
    top_k: int
    query_history: list[str]


@dataclass(frozen=True)
class AgentPlan:
    """Agent 完成检索决策后交给生成层的结果。"""

    question: str
    route: Route
    search_query: str
    hits: tuple[SearchResult, ...]
    retrieval_attempts: int
    rewrite_count: int
    query_history: tuple[str, ...]
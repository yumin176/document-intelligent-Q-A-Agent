"""M5 Agent 工作流的共享状态。"""

from typing import Literal, TypedDict

from app.vectorstore.base import SearchResult

Route = Literal["retrieve", "direct"]


class AgentState(TypedDict):
    """LangGraph 节点之间共享的状态。"""

    question: str
    route: Route
    search_query: str

    candidate_chunks: list[SearchResult]
    relevant_chunks: list[SearchResult]

    retrieval_attempts: int
    rewrite_count: int
    max_rewrites: int
    top_k: int

    query_history: list[str]

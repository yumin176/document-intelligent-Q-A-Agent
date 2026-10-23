"""M5 最小 Agent 状态图。"""

from typing import cast

from langgraph.graph import END, START, StateGraph

from app.agent.grader import grade_chunks
from app.agent.rewrite import rewrite_query
from app.agent.router import route_question
from app.agent.state import AgentState, Route
from app.llm.base import ChatModel
from app.retrieval.base import Retriever


class AgentGraph:
    """Router → Retrieve → Grader → Rewrite 的检索决策图。"""

    def __init__(
        self,
        chat_model: ChatModel,
        retriever: Retriever,
        top_k: int = 4,
        max_rewrites: int = 1,
    ) -> None:
        if top_k <= 0:
            raise ValueError("top_k 必须大于 0")
        if max_rewrites < 0:
            raise ValueError("max_rewrites 不能小于 0")
        self.chat_model = chat_model
        self.retriever = retriever
        self.top_k = top_k
        self.max_rewrites = max_rewrites
        self.graph = self._build_graph()

    def _build_graph(self):
        graph = StateGraph(AgentState)
        graph.add_node("router", self._router_node)
        graph.add_node("retrieve", self._retriever_node)
        graph.add_node("grade", self._grader_node)
        graph.add_node("rewrite", self._rewrite_node)

        graph.add_edge(START, "router")
        graph.add_conditional_edges(
            "router",
            self._route_after_router,
            {"direct": END, "retrieve": "retrieve"},
        )
        graph.add_edge("retrieve", "grade")
        graph.add_conditional_edges(
            "grade",
            self._route_after_grader,
            {"ready": END, "rewrite": "rewrite", "give_up": END},
        )
        graph.add_edge("rewrite", "retrieve")
        return graph.compile()

    def run(self, question: str, top_k: int | None = None) -> AgentState:
        if not question.strip():
            raise ValueError("问题不能为空")
        k = self.top_k if top_k is None else top_k
        if k <= 0:
            raise ValueError("top_k 必须大于 0")

        state = self.graph.invoke(
            {
                "question": question,
                "route": "retrieve",
                "search_query": question,
                "candidate_chunks": [],
                "relevant_chunks": [],
                "retrieval_attempts": 0,
                "rewrite_count": 0,
                "max_rewrites": self.max_rewrites,
                "top_k": k,
                "query_history": [],
            }
        )
        return cast(AgentState, state)

    def route(self, question: str) -> Route:
        return self.run(question)["route"]

    def _router_node(self, state: AgentState) -> dict:
        route = route_question(
            question=state["question"],
            chat_model=self.chat_model,
        )
        return {"route": route}

    @staticmethod
    def _route_after_router(state: AgentState) -> Route:
        route = state.get("route", "retrieve")
        return route if route in {"direct", "retrieve"} else "retrieve"

    def _retriever_node(self, state: AgentState) -> dict:
        query = state["search_query"] or state["question"]
        candidate_chunks = self.retriever.search(
            query=query,
            top_k=state["top_k"] or self.top_k,
        )

        seen: set[str] = set()
        result = []
        for chunk in candidate_chunks:
            if chunk.id in seen:
                continue
            seen.add(chunk.id)
            result.append(chunk)

        return {
            "candidate_chunks": result,
            "retrieval_attempts": state["retrieval_attempts"] + 1,
            "query_history": state["query_history"] + [query],
        }

    def _grader_node(self, state: AgentState) -> dict:
        relevant_chunks = grade_chunks(
            question=state["question"],
            candidates=state["candidate_chunks"],
            chat_model=self.chat_model,
        )
        return {"relevant_chunks": relevant_chunks}

    @staticmethod
    def _route_after_grader(state: AgentState) -> str:
        if state["relevant_chunks"]:
            return "ready"
        if state["retrieval_attempts"] <= state["max_rewrites"]:
            return "rewrite"
        return "give_up"

    def _rewrite_node(self, state: AgentState) -> dict:
        new_query = rewrite_query(
            question=state["question"],
            current_query=state["search_query"],
            candidates=state["candidate_chunks"],
            chat_model=self.chat_model,
        )
        return {
            "search_query": new_query,
            "rewrite_count": state["rewrite_count"] + 1,
            "relevant_chunks": [],
        }
"""Agent 工作流的生成层。"""

from collections.abc import Iterator

from app.agent.graph import AgentRetrievalGraph
from app.agent.state import AgentPlan
from app.llm.base import ChatMessage, ChatModel
from app.prompts.templates import (
    build_direct_answer_prompt,
    build_rag_system_prompt,
)
from app.rag.models import RAGAnswer, Source
from app.rag.service import build_context, build_sources

NO_RELEVANT_ANSWER = (
    "资料中未找到相关信息。你可以尝试补充关键词，"
    "或上传包含相关内容的文档。"
)


class AgentService:
    """执行 Agent 检索决策，再调用 LLM 生成回答。"""

    def __init__(
        self,
        graph: AgentRetrievalGraph,
        chat_model: ChatModel,
        top_k: int = 4,
    ) -> None:
        if top_k <= 0:
            raise ValueError("top_k 必须大于 0")
        self.graph = graph
        self.chat_model = chat_model
        self.top_k = top_k

    def answer(self, question: str, top_k: int | None = None) -> RAGAnswer:
        plan = self._plan(question, top_k)
        sources = self._sources(plan)

        if self._should_refuse(plan):
            return RAGAnswer(answer=NO_RELEVANT_ANSWER, sources=[])

        messages = self._build_messages(plan)
        return RAGAnswer(
            answer=self.chat_model.chat(messages),
            sources=sources,
        )

    def answer_stream(
        self,
        question: str,
        top_k: int | None = None,
    ) -> tuple[Iterator[str], list[Source]]:
        plan = self._plan(question, top_k)
        sources = self._sources(plan)

        if self._should_refuse(plan):
            return iter((NO_RELEVANT_ANSWER,)), []

        messages = self._build_messages(plan)
        return self.chat_model.stream(messages), sources

    def _plan(self, question: str, top_k: int | None) -> AgentPlan:
        k = self.top_k if top_k is None else top_k
        if k <= 0:
            raise ValueError("top_k 必须大于 0")
        return self.graph.plan(question, top_k=k)

    @staticmethod
    def _should_refuse(plan: AgentPlan) -> bool:
        return plan.route == "retrieve" and not plan.hits

    @staticmethod
    def _sources(plan: AgentPlan) -> list[Source]:
        if plan.route == "direct":
            return []
        return build_sources(list(plan.hits))

    @staticmethod
    def _build_messages(plan: AgentPlan) -> list[ChatMessage]:
        if plan.route == "direct":
            return [
                ChatMessage(role="system", content=build_direct_answer_prompt()),
                ChatMessage(role="user", content=plan.question),
            ]

        context = build_context(list(plan.hits))
        return [
            ChatMessage(role="system", content=build_rag_system_prompt(context)),
            ChatMessage(role="user", content=plan.question),
        ]
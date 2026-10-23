"""Agent 回答生成层。"""

from collections.abc import Iterator

from app.agent.graph import AgentGraph
from app.agent.state import AgentState
from app.llm.base import ChatMessage, ChatModel
from app.prompts.templates import build_direct_answer_prompt, build_rag_system_prompt
from app.rag.models import RAGAnswer, Source
from app.rag.service import build_context, build_sources

# 字符串的换行写法，行尾和行首会自动拼接在一起，不是元组类型
NO_RELEVANT_ANSWER = (
    "资料中未找到相关信息。你可以尝试补充关键词，"
    "或上传包含相关内容的文档。"
)


class AgentService:
    """执行 Agent 决策并生成回答。"""

    def __init__(self, graph: AgentGraph, chat_model: ChatModel, top_k: int = 4) -> None:
        if top_k <= 0:
            raise ValueError("top_k 必须大于 0")
        self.graph = graph
        self.chat_model = chat_model
        self.top_k = top_k

    def answer(self, question: str, top_k: int | None = None) -> RAGAnswer:
        """回答普通问题，并返回过滤后的引用。"""
        k = self.top_k if top_k is None else top_k
        state = self.graph.run(question=question, top_k=k)

        if self._should_refuse(state):
            return RAGAnswer(answer=NO_RELEVANT_ANSWER, sources=[])

        return RAGAnswer(
            answer=self.chat_model.chat(self._build_messages(state)),
            sources=self._sources(state),
        )
    def answer_stream(
        self,
        question: str,
        top_k: int | None = None,
    ) -> tuple[Iterator[str], list[Source]]:
        k = self.top_k if top_k is None else top_k
        state = self.graph.run(question=question, top_k=k)

        if self._should_refuse(state):
            return (iter((NO_RELEVANT_ANSWER,)), [])

        return (
            self.chat_model.stream(self._build_messages(state)),
            self._sources(state)
        )


    @staticmethod
    def _should_refuse(state: AgentState) -> bool:
        """retrieve 路由且没有相关 chunk 时返回 True。"""
        return state["route"] == "retrieve" and not state["relevant_chunks"]

    @staticmethod
    def _sources(state: AgentState) -> list[Source]:
        """direct 返回空引用；retrieve 使用过滤后的相关 chunk。"""
        if state["route"] == "direct":
            return []
        return build_sources(state["relevant_chunks"])

    @staticmethod
    def _build_messages(state: AgentState) -> list[ChatMessage]:
        """构造 system + user 消息。"""
        user_msg = ChatMessage(role="user", content=state["question"])

        if state["route"] == "direct":
            system_content = build_direct_answer_prompt()
        else:
            context = build_context(state["relevant_chunks"])
            system_content = build_rag_system_prompt(context)

        return [
            ChatMessage(role="system", content=system_content),
            user_msg,
        ]

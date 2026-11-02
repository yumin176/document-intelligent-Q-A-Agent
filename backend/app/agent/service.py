"""Agent 回答生成层。"""

import time
from collections.abc import Iterator

from app.agent.graph import AgentGraph
from app.agent.state import AgentState
from app.llm.base import ChatMessage, ChatModel
from app.observability.tracing import record_output, trace_generation
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

        messages = self._build_messages(state)
        started = time.perf_counter()
        with trace_generation(self._messages_payload(messages)) as span:
            answer = self.chat_model.chat(messages)
            record_output(
                span,
                answer_chars=len(answer),
                elapsed_ms=round((time.perf_counter() - started) * 1000, 1),
            )

        return RAGAnswer(
            answer=answer,
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

        messages = self._build_messages(state)
        payload = self._messages_payload(messages)

        def _stream() -> Iterator[str]:
            # trace 的生命周期包住整个流式消费过程（与调用方同线程），
            # 记录首 token 延迟与总字数
            started = time.perf_counter()
            first_token_ms: float | None = None
            chars = 0
            with trace_generation(payload, stream=True) as span:
                for token in self.chat_model.stream(messages):
                    if first_token_ms is None:
                        first_token_ms = round(
                            (time.perf_counter() - started) * 1000, 1
                        )
                    chars += len(token)
                    yield token
                record_output(
                    span,
                    answer_chars=chars,
                    first_token_ms=first_token_ms,
                    total_ms=round((time.perf_counter() - started) * 1000, 1),
                )

        return (_stream(), self._sources(state))


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

    @staticmethod
    def _messages_payload(messages: list[ChatMessage]) -> list[dict[str, str]]:
        """把消息序列化为可追踪的字典列表。"""
        return [
            {"role": str(m.role), "content": str(m.content)} for m in messages
        ]

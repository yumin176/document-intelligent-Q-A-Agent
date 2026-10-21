"""RAG 组装服务：检索 + Prompt 组装 + 生成回答。"""

from collections.abc import Iterator

from app.llm.base import ChatMessage, ChatModel
from app.prompts.templates import build_rag_system_prompt
from app.rag.models import RAGAnswer, Source
from app.retrieval.base import Retriever
from app.vectorstore.base import SearchResult


def build_context(hits: list[SearchResult]) -> str:
    """把检索结果格式化为带编号的资料上下文。"""
    context_blocks = []
    for index, hit in enumerate(hits, start=1):
        source_name = (
            hit.metadata.get("filename")
            or hit.metadata.get("source")
            or hit.id
        )
        context_blocks.append(f"[{index}] 来源: {source_name}\n{hit.text}")
    return "\n\n".join(context_blocks)


def build_sources(hits: list[SearchResult]) -> list[Source]:
    """把检索结果转换为 API 使用的引用来源。"""
    return [
        Source(
            chunk_id=hit.id,
            score=hit.score,
            text=hit.text,
            metadata=hit.metadata,
        )
        for hit in hits
    ]


class RAGService:
    """把检索、Prompt 组装和 LLM 生成串联起来。"""

    def __init__(
        self,
        retriever: Retriever,
        chat_model: ChatModel,
        top_k: int = 4,
    ) -> None:
        self.retriever = retriever
        self.chat_model = chat_model
        self.top_k = top_k

    def answer(self, question: str, top_k: int | None = None) -> RAGAnswer:
        """回答问题并返回带引用来源的结果。"""
        k = top_k or self.top_k
        hits = self.retriever.search(question, top_k=k)
        context = build_context(hits)
        messages = self._build_messages(context, question)
        answer_text = self.chat_model.chat(messages)

        return RAGAnswer(
            answer=answer_text,
            sources=build_sources(hits),
        )

    def answer_stream(
        self,
        question: str,
        top_k: int | None = None,
    ) -> tuple[Iterator[str], list[Source]]:
        k = top_k or self.top_k
        hits = self.retriever.search(question, top_k=k)
        context = build_context(hits)
        messages = self._build_messages(context, question)

        return self.chat_model.stream(messages), build_sources(hits)

    @staticmethod
    def _build_context(hits) -> str:
        return build_context(hits)

    @staticmethod
    def _build_messages(context: str, question: str) -> list[ChatMessage]:
        return [
            ChatMessage(role="system", content=build_rag_system_prompt(context)),
            ChatMessage(role="user", content=question),
        ]

    @staticmethod
    def _build_sources(hits) -> list[Source]:
        return build_sources(hits)
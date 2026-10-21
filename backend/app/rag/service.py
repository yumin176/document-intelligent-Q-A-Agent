"""RAG 组装服务：检索 + Prompt 组装 + 生成回答。"""

from collections.abc import Iterator

from app.llm.base import ChatMessage, ChatModel
from app.prompts.templates import build_rag_system_prompt
from app.rag.models import RAGAnswer, Source
from app.retrieval.base import Retriever


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
        context = self._build_context(hits)
        messages = self._build_messages(context, question)
        answer_text = self.chat_model.chat(messages)

        return RAGAnswer(
            answer=answer_text,
            sources=self._build_sources(hits),
        )

    def answer_stream(
        self,
        question: str,
        top_k: int | None = None,
    ) -> tuple[Iterator[str], list[Source]]:
        k = top_k or self.top_k
        hits = self.retriever.search(question, top_k=k)
        context = self._build_context(hits)
        messages = self._build_messages(context, question)

        return self.chat_model.stream(messages), self._build_sources(hits)

    @staticmethod
    def _build_context(hits) -> str:
        context_blocks = []
        for index, hit in enumerate(hits, start=1):
            source_name = (
                hit.metadata.get("filename")
                or hit.metadata.get("source")
                or hit.id
            )
            context_blocks.append(f"[{index}] 来源: {source_name}\n{hit.text}")
        return "\n\n".join(context_blocks)

    @staticmethod
    def _build_messages(context: str, question: str) -> list[ChatMessage]:
        return [
            ChatMessage(role="system", content=build_rag_system_prompt(context)),
            ChatMessage(role="user", content=question),
        ]

    @staticmethod
    def _build_sources(hits) -> list[Source]:
        return [
            Source(
                chunk_id=hit.id,
                score=hit.score,
                text=hit.text,
                metadata=hit.metadata,
            )
            for hit in hits
        ]
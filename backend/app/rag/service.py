"""RAG 组装服务：检索 + Prompt 组装 + 生成回答。"""

from app.embedding.base import Embedder
from app.llm.base import ChatMessage, ChatModel
from app.prompts.templates import build_rag_system_prompt
from app.rag.models import RAGAnswer, Source
from app.vectorstore.base import SearchResult, VectorStore


class RAGService:
    """把 Embedding、向量检索和 LLM 生成串联起来。"""

    def __init__(
        self,
        embedder: Embedder,
        vector_store: VectorStore,
        chat_model: ChatModel,
        top_k: int = 4,
    ) -> None:
        self.embedder = embedder
        self.vector_store = vector_store
        self.chat_model = chat_model
        self.top_k = top_k

    def answer(self, question: str, top_k: int | None = None) -> RAGAnswer:
        """回答问题并返回带引用来源的结果。"""
        k = top_k or self.top_k

        # 1. 问题向量化
        question_vector = self.embedder.embed(question)

        # 2. 向量检索
        hits = self.vector_store.search(question_vector, top_k=k)

        # 3. 组装上下文，并给每段资料编号
        context_blocks = []
        for index, hit in enumerate(hits, start=1):
            source_name = hit.metadata.get("filename") or hit.metadata.get("source") or hit.id
            context_blocks.append(
                f"[{index}] 来源: {source_name}\n{hit.text}"
            )
        context = "\n\n".join(context_blocks)

        # 4. 生成回答
        system_prompt = build_rag_system_prompt(context)
        messages = [
            ChatMessage(role="system", content=system_prompt),
            ChatMessage(role="user", content=question),
        ]
        answer_text = self.chat_model.chat(messages)

        # 5. 组装引用来源
        sources = [
            Source(
                chunk_id=hit.id,
                score=hit.score,
                text=hit.text,
                metadata=hit.metadata,
            )
            for hit in hits
        ]

        return RAGAnswer(answer=answer_text, sources=sources)
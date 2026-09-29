"""LLM 对话模型抽象接口。"""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class ChatMessage:
    """一条对话消息。role 通常为 system / user / assistant。"""

    role: str
    content: str


class ChatModel(ABC):
    """LLM 对话接口。

    上层 RAG 服务只依赖这个接口，后续可替换 OpenAI / 本地模型等实现。
    """

    @abstractmethod
    def chat(
        self,
        messages: list[ChatMessage],
        temperature: float | None = None,
    ) -> str:
        """根据消息列表生成回答。"""

    def complete(self, prompt: str, temperature: float | None = None) -> str:
        """把一段纯文本当作单条 user 消息交给模型。"""
        return self.chat(
            [ChatMessage(role="user", content=prompt)],
            temperature=temperature,
        )
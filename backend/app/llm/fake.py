"""测试用的假 LLM 实现。"""

from app.llm.base import ChatMessage, ChatModel


class FakeChatModel(ChatModel):
    """不调用真实模型，返回固定回答，并记录收到的消息。"""

    def __init__(self, response: str | None = None) -> None:
        self.response = response or "fake answer"
        self.calls: list[list[ChatMessage]] = []

    def chat(
        self,
        messages: list[ChatMessage],
        temperature: float | None = None,
    ) -> str:
        self.calls.append(messages)
        return self.response
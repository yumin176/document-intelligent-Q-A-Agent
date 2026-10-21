"""测试用的假 LLM 实现。"""

from typing import Iterator

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
    def stream(self, messages: list[ChatMessage], temperature: float | None = None) -> Iterator[str]:
        self.calls.append(messages)
        for i in self.response:
            yield i


class ScriptedChatModel(ChatModel):
    """按预设顺序返回响应，用于测试 Agent 多节点调用。"""

    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.calls: list[list[ChatMessage]] = []

    def _next_response(self) -> str:
        if not self.responses:
            raise RuntimeError("ScriptedChatModel 的预设响应已耗尽")
        return self.responses.pop(0)

    def chat(
        self,
        messages: list[ChatMessage],
        temperature: float | None = None,
    ) -> str:
        self.calls.append(messages)
        return self._next_response()

    def stream(
        self,
        messages: list[ChatMessage],
        temperature: float | None = None,
    ) -> Iterator[str]:
        self.calls.append(messages)
        response = self._next_response()
        yield from response
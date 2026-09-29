"""ChatModel 接口与 FakeChatModel 的单元测试。"""

from app.llm.base import ChatMessage
from app.llm.fake import FakeChatModel


def test_fake_chat_returns_response() -> None:
    model = FakeChatModel(response="hello")
    assert model.complete("你好") == "hello"


def test_fake_chat_records_messages() -> None:
    model = FakeChatModel(response="hello")
    model.chat(
        [
            ChatMessage(role="system", content="你是助手"),
            ChatMessage(role="user", content="问题"),
        ]
    )
    assert len(model.calls) == 1
    assert model.calls[0][0].role == "system"
    assert model.calls[0][1].content == "问题"


def test_complete_wraps_prompt_as_user_message() -> None:
    model = FakeChatModel(response="ok")
    model.complete("帮我")
    assert model.calls[0][0].role == "user"
    assert model.calls[0][0].content == "帮我"
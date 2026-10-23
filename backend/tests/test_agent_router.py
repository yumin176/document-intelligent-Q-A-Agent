import pytest

from app.agent.router import heuristic_route, parse_route, route_question
from app.llm.base import ChatMessage, ChatModel
from app.llm.fake import FakeChatModel


class BrokenChatModel(ChatModel):
    def chat(
        self,
        messages: list[ChatMessage],
        temperature: float | None = None,
    ) -> str:
        raise TimeoutError("llm timeout")

    def stream(self, messages: list[ChatMessage], temperature: float | None = None):
        raise TimeoutError("llm timeout")


def test_parse_route_accepts_json_and_markdown() -> None:
    assert parse_route('{"route":"direct"}') == "direct"
    assert parse_route('```json\n{"route":"retrieve"}\n```') == "retrieve"
    assert parse_route('路由结果：{"route":"direct"}') == "direct"


def test_parse_route_rejects_invalid_output() -> None:
    with pytest.raises(ValueError):
        parse_route("not json")
    with pytest.raises(ValueError):
        parse_route('{"route":"unknown"}')


def test_heuristic_route_handles_greetings_and_questions() -> None:
    assert heuristic_route("你好") == "direct"
    assert heuristic_route("你好！") == "direct"
    assert heuristic_route("谢谢！") == "direct"
    assert heuristic_route("var 和 let 有什么区别") == "retrieve"
    assert heuristic_route("你好，请解释一下 var") == "retrieve"


def test_route_question_uses_valid_llm_output() -> None:
    model = FakeChatModel(response='{"route":"direct"}')
    assert route_question("测试问题", model) == "direct"


def test_route_question_falls_back_on_invalid_output() -> None:
    assert route_question("var 是什么", FakeChatModel("bad output")) == "retrieve"
    assert route_question("你好", FakeChatModel("bad output")) == "direct"


def test_route_question_falls_back_on_llm_error() -> None:
    assert route_question("var 是什么", BrokenChatModel()) == "retrieve"
    assert route_question("你好", BrokenChatModel()) == "direct"
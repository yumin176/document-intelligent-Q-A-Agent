import pytest

from app.agent.grader import grade_chunks, parse_relevant_indices
from app.llm.base import ChatMessage, ChatModel
from app.llm.fake import FakeChatModel
from app.vectorstore.base import SearchResult


def make_hits() -> list[SearchResult]:
    return [
        SearchResult(id="a", score=0.9, text="内容 A"),
        SearchResult(id="b", score=0.8, text="内容 B"),
        SearchResult(id="c", score=0.7, text="内容 C"),
    ]


class BrokenChatModel(ChatModel):
    def chat(
        self,
        messages: list[ChatMessage],
        temperature: float | None = None,
    ) -> str:
        raise TimeoutError("llm timeout")

    def stream(self, messages: list[ChatMessage], temperature: float | None = None):
        raise TimeoutError("llm timeout")


def test_parse_relevant_indices_valid_and_deduplicated() -> None:
    assert parse_relevant_indices('{"relevant_indices":[3,1,3]}', 3) == [1, 3]
    assert parse_relevant_indices('```json\n{"relevant_indices":[]}\n```', 3) == []


def test_parse_relevant_indices_rejects_invalid_values() -> None:
    invalid_outputs = [
        "not json",
        '{"relevant_indices":[0]}',
        '{"relevant_indices":[4]}',
        '{"relevant_indices":[true]}',
        '{"relevant_indices":["1"]}',
        '{"relevant_indices":1}',
    ]
    for raw in invalid_outputs:
        with pytest.raises(ValueError):
            parse_relevant_indices(raw, 3)


def test_empty_candidates_do_not_call_llm() -> None:
    model = FakeChatModel('{"relevant_indices":[1]}')
    assert grade_chunks("问题", [], model) == []
    assert model.calls == []


def test_grade_chunks_filters_and_preserves_candidate_order() -> None:
    model = FakeChatModel('{"relevant_indices":[3,1]}')

    result = grade_chunks("问题", make_hits(), model)

    assert [item.id for item in result] == ["a", "c"]


def test_grade_chunks_returns_empty_when_all_irrelevant() -> None:
    model = FakeChatModel('{"relevant_indices":[]}')

    assert grade_chunks("问题", make_hits(), model) == []


def test_grade_chunks_falls_back_to_all_on_invalid_output() -> None:
    model = FakeChatModel("not json")

    result = grade_chunks("问题", make_hits(), model)

    assert [item.id for item in result] == ["a", "b", "c"]


def test_grade_chunks_falls_back_to_all_on_llm_error() -> None:
    result = grade_chunks("问题", make_hits(), BrokenChatModel())

    assert [item.id for item in result] == ["a", "b", "c"]
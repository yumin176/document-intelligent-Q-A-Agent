"""LangSmith 追踪封装的测试。

核心约束：默认配置（.env 无 LANGSMITH_*）下，一切函数都是透传，
绝不发起网络请求。启用路径用替身验证装配逻辑，不碰真实网络。
"""

import pytest

from app.config import Settings
from app.observability import tracing
from app.observability.tracing import (
    is_enabled,
    record_output,
    trace_generation,
    trace_request,
)


@pytest.fixture(autouse=True)
def _reset_cache():
    tracing.reset_client_cache()
    yield
    tracing.reset_client_cache()


class TestDisabled:
    """未启用追踪时的一切行为。"""

    def test_default_settings_disables_tracing(self) -> None:
        assert is_enabled() is False

    def test_api_key_alone_is_not_enough(self, monkeypatch) -> None:
        settings = Settings(langsmith_tracing=False, langsmith_api_key="sk-x")
        monkeypatch.setattr(tracing, "get_settings", lambda: settings)
        tracing.reset_client_cache()
        assert is_enabled() is False

    def test_trace_request_yields_none(self) -> None:
        with trace_request("问题", 4, "req-1") as root:
            assert root is None

    def test_trace_generation_yields_none(self) -> None:
        with trace_generation([{"role": "user", "content": "hi"}]) as span:
            assert span is None

    def test_record_output_none_is_noop(self) -> None:
        record_output(None, answer_chars=3)  # 不应抛错


class TestScrub:
    """载荷脱敏：去掉 chunk 正文，避免 trace 膨胀。"""

    def test_removes_chunk_text(self) -> None:
        payload = {
            "question": "营收多少",
            "relevant_chunks": [
                {"id": "doc#1:abc", "score": 0.9, "text": "很长的正文" * 100}
            ],
        }
        cleaned = tracing._scrub_run_data(payload)
        assert cleaned["relevant_chunks"] == [{"id": "doc#1:abc", "score": 0.9}]
        assert "text" not in str(cleaned)

    def test_handles_search_result_objects(self) -> None:
        """LangGraph 节点返回的是 SearchResult 对象，必须同样压成简报。"""

        class _Chunk:
            def __init__(self) -> None:
                self.id = "doc#2:def"
                self.score = 0.8
                self.text = "正文" * 500

        cleaned = tracing._scrub_run_data({"candidate_chunks": [_Chunk()]})
        assert cleaned["candidate_chunks"] == [{"id": "doc#2:def", "score": 0.8}]

    def test_keeps_small_fields(self) -> None:
        payload = {"route": "retrieve", "top_k": 3, "nested": {"answer_chars": 62}}
        assert tracing._scrub_run_data(payload) == payload

    def test_handles_non_dict(self) -> None:
        assert tracing._scrub_run_data("纯文本") == "纯文本"
        assert tracing._scrub_run_data(None) is None


class _FakeRun:
    def __init__(self) -> None:
        self.outputs: dict = {}


class _FakeTrace:
    """替身 trace：捕获参数并 yield 一个可写 outputs 的对象。"""

    last_kwargs: dict = {}
    run = _FakeRun()

    def __init__(self, **kwargs) -> None:
        type(self).last_kwargs = kwargs

    def __enter__(self):
        return type(self).run

    def __exit__(self, *exc):
        return False


class TestEnabled:
    """启用路径：用替身验证参数装配，不发起网络请求。"""

    @pytest.fixture
    def fake_client(self, monkeypatch):
        settings = Settings(
            langsmith_tracing=True,
            langsmith_api_key="sk-test",
            langsmith_project="proj-test",
            langsmith_endpoint="http://127.0.0.1:1",
        )
        monkeypatch.setattr(tracing, "get_settings", lambda: settings)
        monkeypatch.setattr(tracing, "trace", _FakeTrace)
        tracing.reset_client_cache()

        class _Client:
            pass

        monkeypatch.setattr(
            tracing, "Client", lambda **kwargs: _Client()  # noqa: ARG005
        )
        tracing.reset_client_cache()
        yield

    def test_enabled(self, fake_client) -> None:
        assert is_enabled() is True

    def test_request_metadata(self, fake_client) -> None:
        _FakeTrace.last_kwargs = {}
        with trace_request("财报问题", 5, "abc123") as run:
            assert run is _FakeTrace.run
        kwargs = _FakeTrace.last_kwargs
        assert kwargs["name"] == "AgentService.answer"
        assert kwargs["inputs"] == {"question": "财报问题", "top_k": 5}
        assert kwargs["metadata"]["request_id"] == "abc123"
        assert kwargs["metadata"]["retrieval_mode"] == "hybrid_rerank"

    def test_generation_span(self, fake_client) -> None:
        _FakeTrace.last_kwargs = {}
        with trace_generation([{"role": "user", "content": "hi"}], stream=True):
            pass
        kwargs = _FakeTrace.last_kwargs
        assert kwargs["name"] == "generate"
        assert kwargs["run_type"] == "llm"
        assert kwargs["inputs"]["stream"] is True
        # 模型名来自配置（.env 可能覆盖代码默认值），不写死
        assert kwargs["metadata"]["llm_model"] == Settings().llm_model

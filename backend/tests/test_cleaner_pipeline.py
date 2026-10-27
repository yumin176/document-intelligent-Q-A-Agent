from dataclasses import FrozenInstanceError
from hashlib import sha256

import pytest

from app.ingestion.cleaning import BaseCleaner, CleanerPipeline


def digest(text: str) -> str:
    return sha256(text.encode("utf-8")).hexdigest()


class UpperCleaner(BaseCleaner):
    name = "upper"

    def clean(self, text: str) -> str:
        return text.upper()


class TruncateCleaner(BaseCleaner):
    name = "truncate"

    def clean(self, text: str) -> str:
        return text[:3]


class NoOpCleaner(BaseCleaner):
    name = "noop"

    def clean(self, text: str) -> str:
        return text


class NoneCleaner(BaseCleaner):
    name = "none"

    def clean(self, text: str) -> str:
        return None  # type: ignore[return-value]


class BlankNameCleaner(BaseCleaner):
    name = ""

    def clean(self, text: str) -> str:
        return text


class ExplodingCleaner(BaseCleaner):
    name = "exploding"

    def clean(self, text: str) -> str:
        raise RuntimeError("cleaner failed")


def test_empty_pipeline_returns_original_text_without_steps() -> None:
    text = "abcdefg"

    result = CleanerPipeline([]).clean(text)

    assert result.text == text
    assert result.steps == ()


def test_pipeline_runs_cleaners_in_the_declared_order() -> None:
    pipeline = CleanerPipeline([UpperCleaner(), TruncateCleaner()])

    result = pipeline.clean("abcdefg")

    assert result.text == "ABC"
    assert tuple(step.cleaner_name for step in result.steps) == ("upper", "truncate")


def test_pipeline_records_lengths_and_hashes_for_each_step() -> None:
    pipeline = CleanerPipeline([UpperCleaner(), TruncateCleaner()])

    result = pipeline.clean("abcdefg")

    first, second = result.steps
    assert first.before_chars == 7
    assert first.after_chars == 7
    assert first.before_sha256 == digest("abcdefg")
    assert first.after_sha256 == digest("ABCDEFG")
    assert first.before_sha256 != first.after_sha256

    assert second.before_chars == 7
    assert second.after_chars == 3
    assert second.before_sha256 == digest("ABCDEFG")
    assert second.after_sha256 == digest("ABC")


def test_pipeline_hashes_unicode_as_utf8() -> None:
    result = CleanerPipeline([NoOpCleaner()]).clean("你好")

    assert result.steps[0].before_sha256 == digest("你好")
    assert result.steps[0].after_sha256 == digest("你好")


def test_pipeline_records_noop_cleaners() -> None:
    result = CleanerPipeline([NoOpCleaner()]).clean("same text")

    assert result.text == "same text"
    assert len(result.steps) == 1
    assert result.steps[0].before_chars == len("same text")
    assert result.steps[0].before_sha256 == result.steps[0].after_sha256


def test_result_and_steps_are_immutable() -> None:
    result = CleanerPipeline([UpperCleaner()]).clean("abc")

    assert isinstance(result.steps, tuple)
    with pytest.raises(FrozenInstanceError):
        result.text = "changed"  # type: ignore[misc]
    with pytest.raises(AttributeError):
        result.steps.append(result.steps[0])  # type: ignore[attr-defined]


def test_pipeline_copies_cleaner_sequence_at_construction() -> None:
    cleaners = [UpperCleaner()]
    pipeline = CleanerPipeline(cleaners)

    cleaners.append(TruncateCleaner())

    result = pipeline.clean("abcdefg")
    assert result.text == "ABCDEFG"
    assert len(result.steps) == 1


def test_pipeline_rejects_duplicate_cleaner_names_at_construction() -> None:
    with pytest.raises(ValueError, match="重复"):
        CleanerPipeline([UpperCleaner(), UpperCleaner()])


def test_pipeline_rejects_blank_cleaner_names_at_construction() -> None:
    with pytest.raises(ValueError, match="不能为空"):
        CleanerPipeline([BlankNameCleaner()])


def test_pipeline_rejects_objects_that_are_not_cleaners() -> None:
    with pytest.raises(TypeError, match="BaseCleaner"):
        CleanerPipeline([object()])  # type: ignore[list-item]


def test_clean_rejects_non_string_input() -> None:
    with pytest.raises(TypeError, match="字符串"):
        CleanerPipeline([]).clean(None)  # type: ignore[arg-type]


def test_clean_rejects_non_string_cleaner_output() -> None:
    with pytest.raises(TypeError, match="非字符串"):
        CleanerPipeline([NoneCleaner()]).clean("abc")


def test_cleaner_exception_is_propagated() -> None:
    with pytest.raises(RuntimeError, match="cleaner failed"):
        CleanerPipeline([ExplodingCleaner()]).clean("abc")
import re

import pytest

from app.ingestion.cleaning import CleanerPipeline, NoiseCleaner, NormalizeCleaner
def test_noise_cleaner():
    raw = "正文\n第 3 页\n这是机密项目的说明。\n机密\n下一段"

    expected = "正文\n这是机密项目的说明。\n下一段"
    cleaner=NoiseCleaner()

    res=cleaner.clean(raw)
    assert res==expected
@pytest.mark.parametrize(
    "page_line",
    [
        "第 3 页",
        "第3页",
        "第 3 页 / 共 10 页",
        "Page 3",
        "Page 3 of 10",
        "P. 3",
        "P3",
        "- 3 -",
        "— 3 —",
    ],
)
def test_common_page_number_lines_are_removed(page_line: str) -> None:
    raw = f"正文\n{page_line}\n下一段"

    assert NoiseCleaner().clean(raw) == "正文\n下一段"


def test_standalone_numeric_line_is_preserved() -> None:
    raw = "正文\n3\n下一段"

    assert NoiseCleaner().clean(raw) == raw


def test_page_number_words_inside_a_sentence_are_preserved() -> None:
    raw = "第 3 页介绍了变量提升。\nPage 3 explains hoisting."

    assert NoiseCleaner().clean(raw) == raw


@pytest.mark.parametrize(
    "watermark",
    ["机密", "内部资料", "仅供内部使用", "CONFIDENTIAL", "confidential"],
)
def test_watermark_lines_are_removed(watermark: str) -> None:
    raw = f"正文\n{watermark}\n下一段"

    assert NoiseCleaner().clean(raw) == "正文\n下一段"


def test_watermark_word_inside_sentence_is_preserved() -> None:
    raw = "这是机密项目的设计说明。"

    assert NoiseCleaner().clean(raw) == raw


def test_code_regions_are_not_treated_as_noise() -> None:
    raw = (
        "正文\n"
        "```python\n"
        "# 第 3 页\n"
        "CONFIDENTIAL = True\n"
        "```\n"
        "`机密`\n"
        "下一段"
    )

    assert NoiseCleaner().clean(raw) == raw


def test_custom_pattern_removes_only_a_full_line() -> None:
    cleaner = NoiseCleaner(extra_patterns=(r"公司内部资料",))
    raw = "公司内部资料\n这是公司内部资料说明\n正文"

    assert cleaner.clean(raw) == "这是公司内部资料说明\n正文"


def test_compiled_custom_pattern_is_supported() -> None:
    cleaner = NoiseCleaner(extra_patterns=(re.compile(r"company header", re.IGNORECASE),))
    raw = "COMPANY HEADER\n正文"

    assert cleaner.clean(raw) == "正文"


def test_invalid_custom_pattern_is_rejected_at_construction() -> None:
    with pytest.raises(ValueError, match="无效正则"):
        NoiseCleaner(extra_patterns=(r"[",))


def test_single_string_custom_pattern_is_rejected() -> None:
    with pytest.raises(TypeError, match="正则列表"):
        NoiseCleaner(extra_patterns=r"公司内部资料")  # type: ignore[arg-type]


def test_builtin_rule_switches_are_independent() -> None:
    raw = "正文\n第 3 页\n机密\n下一段"

    without_page_numbers = NoiseCleaner(remove_page_numbers=False)
    without_watermarks = NoiseCleaner(remove_watermarks=False)

    assert without_page_numbers.clean(raw) == "正文\n第 3 页\n下一段"
    assert without_watermarks.clean(raw) == "正文\n机密\n下一段"


def test_preserves_crlf_and_trailing_newline_for_kept_lines() -> None:
    raw = "正文\r\n第 3 页\r\n下一段\r\n"

    assert NoiseCleaner().clean(raw) == "正文\r\n下一段\r\n"


def test_empty_string_returns_empty_string() -> None:
    assert NoiseCleaner().clean("") == ""


def test_noise_cleaning_is_idempotent() -> None:
    raw = "正文\n第 3 页\n机密\n下一段"
    cleaner = NoiseCleaner()

    cleaned_once = cleaner.clean(raw)
    cleaned_twice = cleaner.clean(cleaned_once)

    assert cleaned_twice == cleaned_once


def test_noise_cleaner_works_inside_pipeline() -> None:
    pipeline = CleanerPipeline([NormalizeCleaner(), NoiseCleaner()])
    raw = "正文\r\n第 ３ 页\r\n下一段"

    result = pipeline.clean(raw)

    assert result.text == "正文\n下一段"
    assert [step.cleaner_name for step in result.steps] == ["normalize", "noise"]
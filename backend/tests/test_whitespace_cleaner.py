import pytest

from app.ingestion.cleaning import CleanerPipeline
from app.ingestion.cleaning.normalize import NormalizeCleaner
from app.ingestion.cleaning.whitespace import WhitespaceCleaner


def test_collapses_multiple_spaces_in_normal_text() -> None:
    assert WhitespaceCleaner().clean("A    B\t\tC") == "A B C"


def test_removes_trailing_spaces_but_preserves_line_indentation() -> None:
    raw = "    indented   text   \n  list item   "

    assert WhitespaceCleaner().clean(raw) == "    indented text\n  list item"


def test_collapses_excess_blank_lines_by_default() -> None:
    assert WhitespaceCleaner().clean("A\n\n\n\nB") == "A\n\nB"


def test_zero_max_blank_lines_keeps_single_line_breaks() -> None:
    cleaner = WhitespaceCleaner(max_blank_lines=0)

    assert cleaner.clean("A\n\n\nB") == "A\nB"
    assert cleaner.clean("A\n") == "A\n"


def test_max_blank_lines_can_be_increased() -> None:
    cleaner = WhitespaceCleaner(max_blank_lines=2)

    assert cleaner.clean("A\n\n\n\n\nB") == "A\n\n\nB"


def test_invalid_max_blank_lines_is_rejected() -> None:
    with pytest.raises(ValueError, match="不能小于 0"):
        WhitespaceCleaner(max_blank_lines=-1)

    with pytest.raises(TypeError, match="整数"):
        WhitespaceCleaner(max_blank_lines=1.5)  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="整数"):
        WhitespaceCleaner(max_blank_lines=True)  # type: ignore[arg-type]


def test_preserves_fenced_code_exactly() -> None:
    raw = "正文   内容\n```js\nconst  a =  1;\n\n\n// 保留空白\n```"

    expected = "正文 内容\n```js\nconst  a =  1;\n\n\n// 保留空白\n```"
    assert WhitespaceCleaner().clean(raw) == expected


def test_preserves_inline_code_exactly() -> None:
    raw = "调用   `fn(  a,   b )`   获取结果"

    assert WhitespaceCleaner().clean(raw) == "调用 `fn(  a,   b )` 获取结果"


def test_does_not_concatenate_text_and_following_code_block() -> None:
    raw = "正文   \n```js\nconst a =  1;\n```"

    expected = "正文\n```js\nconst a =  1;\n```"
    assert WhitespaceCleaner().clean(raw) == expected


def test_does_not_concatenate_code_block_and_following_text() -> None:
    raw = "```js\nconst a =  1;\n```\n后续   正文"

    assert WhitespaceCleaner().clean(raw) == "```js\nconst a =  1;\n```\n后续 正文"


def test_preserves_single_trailing_newline() -> None:
    assert WhitespaceCleaner().clean("A\n") == "A\n"


def test_empty_string_returns_empty_string() -> None:
    assert WhitespaceCleaner().clean("") == ""


def test_whitespace_cleaning_is_idempotent() -> None:
    cleaner = WhitespaceCleaner()
    raw = "正文   内容\n\n\n```js\nconst  a =  1;\n```\n后续   `x  y`"

    cleaned_once = cleaner.clean(raw)
    cleaned_twice = cleaner.clean(cleaned_once)

    assert cleaned_twice == cleaned_once


def test_whitespace_cleaner_works_inside_pipeline() -> None:
    pipeline = CleanerPipeline([NormalizeCleaner(), WhitespaceCleaner()])

    result = pipeline.clean("ＡＰＩ   文档\r\n\r\n\r\n正文")

    assert result.text == "API 文档\n\n正文"
    assert [step.cleaner_name for step in result.steps] == ["normalize", "whitespace"]
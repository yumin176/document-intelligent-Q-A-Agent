import pytest

from app.ingestion.cleaning import (
    CleanerPipeline,
    DedupCleaner,
    NormalizeCleaner,
    WhitespaceCleaner,
)
def test_dedup_cleaner():
    raw = "这是需要保留的完整段落。\n\n这是需要保留的完整段落。\n\n下一段"

    expected = "这是需要保留的完整段落。\n\n下一段"

    cleaner=DedupCleaner(min_chars=1)
    # print(raw)
    # print("*"*30)
    # print(cleaner.clean(raw))
    assert cleaner.clean(raw) == expected

def test_dedup_cleaner2():
    raw = "这是需要保留的完整段落。\n\n这是需要保留的完整段落。\n\n下一段\n\n```python\nprint('Ａ')\n```\n\n```python\nprint('Ａ')\n```"

    expected = "这是需要保留的完整段落。\n\n下一段\n\n```python\nprint('Ａ')\n```\n\n```python\nprint('Ａ')\n```"

    cleaner=DedupCleaner(min_chars=1)
    res=cleaner.clean(raw)
    # print(raw)
    # print("*"*30)
    # print(res)
    assert res == expected
def test_adjacent_duplicate_ignoring_whitespace_differences() -> None:
    first = "这是需要保留的完整段落， 用来说明问题。"
    second = "这是需要保留的完整段落，   用来说明问题。"

    raw = f"{first}\n\n{second}\n\n下一段"

    assert DedupCleaner(min_chars=1).clean(raw) == f"{first}\n\n下一段"


def test_non_adjacent_duplicates_are_kept_by_default() -> None:
    repeated = "这是需要保留的完整段落，用来说明问题。"
    raw = f"{repeated}\n\n中间段落\n\n{repeated}"

    assert DedupCleaner(min_chars=1).clean(raw) == raw


def test_document_scope_removes_non_adjacent_duplicates() -> None:
    repeated = "这是需要保留的完整段落，用来说明问题。"
    raw = f"{repeated}\n\n中间段落\n\n{repeated}"
    expected = f"{repeated}\n\n中间段落"

    cleaner = DedupCleaner(min_chars=1, scope="document")
    assert cleaner.clean(raw) == expected


def test_three_adjacent_duplicates_keep_only_first() -> None:
    paragraph = "这是连续出现三次的完整段落。"
    raw = f"{paragraph}\n\n{paragraph}\n\n{paragraph}"

    assert DedupCleaner(min_chars=1).clean(raw) == paragraph


def test_short_paragraph_breaks_adjacency() -> None:
    paragraph = "这是需要保留的完整段落，用来说明问题。"
    raw = f"{paragraph}\n\n短\n\n{paragraph}"

    assert DedupCleaner(min_chars=10).clean(raw) == raw


def test_duplicate_headings_are_preserved() -> None:
    raw = "## 示例\n\n## 示例"

    assert DedupCleaner(min_chars=1).clean(raw) == raw


def test_indented_duplicate_headings_are_preserved() -> None:
    raw = "   ## 示例\n\n   ## 示例"

    assert DedupCleaner(min_chars=1).clean(raw) == raw


def test_duplicate_paragraph_with_inline_code_is_removed() -> None:
    paragraph = "调用 `fn(  a )` 完成操作。"
    raw = f"{paragraph}\n\n{paragraph}"

    assert DedupCleaner(min_chars=1).clean(raw) == paragraph


def test_inline_code_only_paragraphs_are_preserved() -> None:
    raw = "`same()`\n\n`same()`"

    assert DedupCleaner(min_chars=0).clean(raw) == raw


def test_invalid_scope_is_rejected() -> None:
    with pytest.raises(ValueError, match="scope"):
        DedupCleaner(scope="invalid")  # type: ignore[arg-type]


def test_invalid_min_chars_is_rejected() -> None:
    with pytest.raises(ValueError, match="不能小于 0"):
        DedupCleaner(min_chars=-1)

    with pytest.raises(TypeError, match="整数"):
        DedupCleaner(min_chars=1.5)  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="整数"):
        DedupCleaner(min_chars=True)  # type: ignore[arg-type]


def test_dedup_is_idempotent() -> None:
    raw = "重复段落内容足够长。\n\n重复段落内容足够长。\n\n下一段"
    cleaner = DedupCleaner(min_chars=1)

    cleaned_once = cleaner.clean(raw)
    cleaned_twice = cleaner.clean(cleaned_once)

    assert cleaned_twice == cleaned_once


def test_dedup_works_inside_pipeline() -> None:
    pipeline = CleanerPipeline(
        [
            NormalizeCleaner(),
            WhitespaceCleaner(),
            DedupCleaner(min_chars=1),
        ]
    )
    raw = "第一段   需要保留。\r\n\r\n第一段   需要保留。\r\n\r\n第二段。"

    result = pipeline.clean(raw)

    assert result.text == "第一段 需要保留。\n\n第二段。"
    assert [step.cleaner_name for step in result.steps] == [
        "normalize",
        "whitespace",
        "dedup",
    ]
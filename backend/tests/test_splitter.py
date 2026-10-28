"""split_text 分块函数的单元测试。"""

import pytest

from app.ingestion.splitter import split_text, split_text_fixed, split_text_recursive


def test_split_without_overlap() -> None:
    assert split_text("abcdefghij", chunk_size=5, chunk_overlap=0) == ["abcde", "fghij"]


def test_split_with_overlap() -> None:
    assert split_text("abcdefghij", chunk_size=5, chunk_overlap=2) == ["abcde", "defgh", "ghij"]


def test_short_text_returns_single_chunk() -> None:
    assert split_text("hello", chunk_size=10, chunk_overlap=2) == ["hello"]


def test_empty_text_returns_empty_list() -> None:
    assert split_text("", chunk_size=5, chunk_overlap=2) == []


def test_overlap_equal_to_size_raises() -> None:
    with pytest.raises(ValueError):
        split_text("abcdefghij", chunk_size=5, chunk_overlap=5)


def test_negative_overlap_raises() -> None:
    with pytest.raises(ValueError):
        split_text("abcdefghij", chunk_size=5, chunk_overlap=-1)
def test_recursive_short_text_returns_single_chunk() -> None:
    assert split_text_recursive("hello", chunk_size=10, chunk_overlap=2) == ["hello"]


def test_recursive_empty_text_returns_empty_list() -> None:
    assert split_text_recursive("", chunk_size=10, chunk_overlap=2) == []


def test_recursive_respects_chunk_size_and_is_stable() -> None:
    text = "第一句话很短。第二句话也很短。第三句话同样短。" * 5

    chunks = split_text_recursive(text, chunk_size=20, chunk_overlap=3)

    assert chunks
    assert all(len(chunk) <= 20 for chunk in chunks)
    assert chunks == split_text_recursive(text, chunk_size=20, chunk_overlap=3)


def test_recursive_keeps_chinese_punctuation_with_preceding_sentence() -> None:
    text = "第一句话很短。第二句话也很短。第三句话同样短。"

    chunks = split_text_recursive(text, chunk_size=15, chunk_overlap=3)

    assert all(not chunk.startswith(("。", "！", "？", "；")) for chunk in chunks)
    assert "第一句话很短。" in "".join(chunks)
    assert "第三句话同样短。" in "".join(chunks)


def test_recursive_keeps_english_sentences_readable() -> None:
    text = "First sentence. Second sentence. Third sentence."

    chunks = split_text_recursive(text, chunk_size=20, chunk_overlap=5)

    assert all(not chunk.startswith((". ", "! ", "? ", "; ")) for chunk in chunks)
    assert any("First sentence." in chunk for chunk in chunks)
    assert any("Third sentence." in chunk for chunk in chunks)


def test_recursive_splits_text_without_punctuation() -> None:
    text = "abcdefghijklmnopqrstuvwxyz"

    chunks = split_text_recursive(text, chunk_size=10, chunk_overlap=2)

    assert chunks
    assert all(len(chunk) <= 10 for chunk in chunks)
    for left, right in zip(chunks, chunks[1:]):
        assert left[-2:] == right[:2]


def test_recursive_without_overlap_has_no_overlap() -> None:
    text = "abcdefghijklmnopqrstuvwxyz"

    chunks = split_text_recursive(text, chunk_size=10, chunk_overlap=0)

    assert chunks == ["abcdefghij", "klmnopqrst", "uvwxyz"]


def test_recursive_handles_mixed_chinese_and_english() -> None:
    text = "这是中文句子。This is an English sentence. 再回到中文。"

    chunks = split_text_recursive(text, chunk_size=30, chunk_overlap=4)

    assert chunks
    assert all(len(chunk) <= 30 for chunk in chunks)
    assert any("中文句子" in chunk for chunk in chunks)
    assert any("English sentence" in chunk for chunk in chunks)


def test_strategy_dispatch_to_recursive() -> None:
    text = "abcdefghijklmnopqrstuvwxyz"

    assert split_text(
        text,
        chunk_size=10,
        chunk_overlap=2,
        strategy="recursive",
    ) == split_text_recursive(text, chunk_size=10, chunk_overlap=2)


def test_unknown_strategy_raises() -> None:
    with pytest.raises(ValueError, match="不支持的分块策略"):
        split_text("text", strategy="unknown")  # type: ignore[arg-type]


def test_non_positive_chunk_size_raises_for_both_strategies() -> None:
    with pytest.raises(ValueError, match="chunk_size"):
        split_text_fixed("text", chunk_size=0, chunk_overlap=0)
    with pytest.raises(ValueError, match="chunk_size"):
        split_text_recursive("text", chunk_size=0, chunk_overlap=0)
"""split_text 分块函数的单元测试。"""

import pytest

from app.ingestion.splitter import split_text


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
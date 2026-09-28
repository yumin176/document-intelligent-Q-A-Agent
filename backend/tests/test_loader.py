"""load_document 文档解析函数的单元测试。"""

from pathlib import Path
from unittest.mock import patch

import pytest

from app.ingestion.loader import load_document


def test_load_txt(tmp_path: Path) -> None:
    path = tmp_path / "sample.txt"
    path.write_text("你好，世界", encoding="utf-8")

    content, meta = load_document(path)

    assert content == "你好，世界"
    assert meta["filename"] == "sample.txt"
    assert meta["file_type"] == "txt"
    assert meta["num_chars"] == 5


def test_load_markdown(tmp_path: Path) -> None:
    path = tmp_path / "sample.md"
    path.write_text("# 标题\n正文内容", encoding="utf-8")

    content, meta = load_document(path)

    assert content == "# 标题\n正文内容"
    assert meta["file_type"] == "md"


def test_load_pdf(tmp_path: Path) -> None:
    path = tmp_path / "sample.pdf"
    path.write_bytes(b"fake pdf bytes")

    class FakePage:
        def extract_text(self) -> str:
            return "Page one"

    class FakeReader:
        def __init__(self, _path: str) -> None:
            self.pages = [FakePage(), FakePage()]

    with patch("app.ingestion.loader.PdfReader", return_value=FakeReader("ignored")):
        content, meta = load_document(path)

    assert content == "Page one\nPage one"
    assert meta["file_type"] == "pdf"


def test_unsupported_type_raises(tmp_path: Path) -> None:
    path = tmp_path / "sample.docx"
    path.write_text("not supported", encoding="utf-8")

    with pytest.raises(ValueError):
        load_document(path)


def test_missing_file_raises(tmp_path: Path) -> None:
    path = tmp_path / "not_exist.txt"

    with pytest.raises(ValueError):
        load_document(path)
"""PDF 文本的视觉换行重建工具。"""

import re
from collections.abc import Sequence

from app.ingestion.loader import ParsedPage

SENTENCE_END = set("。！？；：.!?;:")
CLOSING_CHARS = set("”’」』】）)]}")
STRUCTURE_START = re.compile(
    r"^\s*(?:"
    r"第[一二三四五六七八九十百0-9]+[章节部分]|"
    r"[（(]?[0-9一二三四五六七八九十]{1,3}[)）.、]|"
    r"[-*•·]|"
    r"[#|]"
    r")"
)


def _ends_sentence(line: str) -> bool:
    stripped = line.rstrip()
    while stripped and stripped[-1] in CLOSING_CHARS:
        stripped = stripped[:-1]
    return bool(stripped) and stripped[-1] in SENTENCE_END


def _starts_structure(line: str) -> bool:
    return bool(STRUCTURE_START.match(line))


def _join_lines(previous: str, current: str) -> str:
    """中文断行直接拼接，英文单词断行补一个空格。"""
    if not previous:
        return current
    if previous[-1].isascii() and previous[-1].isalnum() and current[0].isascii() and current[0].isalnum():
        return previous + " " + current
    return previous + current


def dewrap_page_text(text: str) -> str:
    """把 PDF 视觉换行合并成段落，保留标题、列表和表格边界。"""
    paragraphs: list[str] = []
    current: list[str] = []

    def flush() -> None:
        if current:
            paragraphs.append("".join(current))
            current.clear()

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            flush()
            continue
        if not current:
            current.append(line)
            continue
        if _ends_sentence(current[-1]) or _starts_structure(line):
            flush()
            current.append(line)
            continue
        current[-1] = _join_lines(current[-1], line)

    flush()
    return "\n\n".join(paragraphs)


def dewrap_pdf_pages(pages: Sequence[ParsedPage]) -> tuple[ParsedPage, ...]:
    return tuple(
        ParsedPage(
            page_number=page.page_number,
            text=dewrap_page_text(page.text),
        )
        for page in pages
    )


def join_pdf_pages(pages: Sequence[ParsedPage]) -> str:
    return "\n\n".join(page.text.rstrip("\r\n") for page in pages)
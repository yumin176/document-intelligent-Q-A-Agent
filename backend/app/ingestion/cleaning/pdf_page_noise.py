import math
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

from app.ingestion.loader import ParsedPage


@dataclass(frozen=True)
class PageCleanResult:
    pages: tuple[ParsedPage, ...]
    removed_header_keys: tuple[str, ...] = ()
    removed_footer_keys: tuple[str, ...] = ()
    removed_page_numbers: int = 0

    @property
    def text(self) -> str:
        # 每页之间只保留一个换行，避免页面文本自带的换行造成额外空行。
        return "\n".join(page.text.rstrip("\r\n") for page in self.pages)


class PdfPageNoiseCleaner:
    """基于 PDF 页面边界识别并删除重复页眉、页脚和页码。"""

    PAGE_NUMBER_PATTERNS = (
        re.compile(r"\d+\s*[/／]\s*\d+"),
        re.compile(r"第\s*\d+\s*页(?:\s*[/／]\s*共\s*\d+\s*页)?"),
        re.compile(r"Page\s+\d+(?:\s+of\s+\d+)?", re.IGNORECASE),
        re.compile(r"P\.?\s*\d+", re.IGNORECASE),
        re.compile(r"[-—–]\s*\d+\s*[-—–]"),
    )

    def __init__(
        self,
        *,
        top_lines: int = 2,
        bottom_lines: int = 2,
        min_pages: int = 5,
        min_count: int = 5,
        min_ratio: float = 0.6,
        min_chars: int = 8,
        max_chars: int = 80,
        remove_page_numbers: bool = True,
    ) -> None:
        self._validate_positive_int("top_lines", top_lines)
        self._validate_positive_int("bottom_lines", bottom_lines)
        self._validate_positive_int("min_pages", min_pages)
        self._validate_positive_int("min_count", min_count)
        self._validate_positive_int("min_chars", min_chars)
        self._validate_positive_int("max_chars", max_chars)

        if min_chars > max_chars:
            raise ValueError("min_chars 不能大于 max_chars")
        if isinstance(min_ratio, bool) or not isinstance(min_ratio, (int, float)):
            raise TypeError("min_ratio 必须是数字")
        if not 0 < min_ratio <= 1:
            raise ValueError("min_ratio 必须位于 (0, 1]")
        if not isinstance(remove_page_numbers, bool):
            raise TypeError("remove_page_numbers 必须是布尔值")

        self.top_lines = top_lines
        self.bottom_lines = bottom_lines
        self.min_pages = min_pages
        self.min_count = min_count
        self.min_ratio = float(min_ratio)
        self.min_chars = min_chars
        self.max_chars = max_chars
        self.remove_page_numbers = remove_page_numbers

    def clean_pages(
        self,
        pages: Sequence[ParsedPage],
    ) -> PageCleanResult:
        pages = tuple(pages)
        if not pages:
            return PageCleanResult(pages=())

        top_counts, bottom_counts = self._count_boundary_lines(pages)
        threshold = max(
            self.min_count,
            math.ceil(len(pages) * self.min_ratio),
        )

        if len(pages) >= self.min_pages:
            header_keys = {
                key for key, count in top_counts.items() if count >= threshold
            }
            footer_keys = {
                key for key, count in bottom_counts.items() if count >= threshold
            }
        else:
            header_keys = set()
            footer_keys = set()

        cleaned_pages: list[ParsedPage] = []
        removed_page_numbers = 0

        for page in pages:
            cleaned_text, removed_count = self._clean_page(
                page.text,
                header_keys,
                footer_keys,
            )
            removed_page_numbers += removed_count
            cleaned_pages.append(
                ParsedPage(
                    page_number=page.page_number,
                    text=cleaned_text,
                )
            )

        return PageCleanResult(
            pages=tuple(cleaned_pages),
            removed_header_keys=tuple(sorted(header_keys)),
            removed_footer_keys=tuple(sorted(footer_keys)),
            removed_page_numbers=removed_page_numbers,
        )

    def _count_boundary_lines(
        self,
        pages: Sequence[ParsedPage],
    ) -> tuple[Counter[str], Counter[str]]:
        top_counts: Counter[str] = Counter()
        bottom_counts: Counter[str] = Counter()

        for page in pages:
            lines = [
                line.strip()
                for line in page.text.splitlines()
                if line.strip()
            ]
            # 过短的页面不参与重复页眉页脚推断，避免顶部和底部区域重叠。
            if len(lines) <= self.top_lines + self.bottom_lines:
                continue

            top_keys = {
                self._normalize(line)
                for line in lines[:self.top_lines]
                if self._is_boundary_candidate(line)
            }
            bottom_keys = {
                self._normalize(line)
                for line in lines[-self.bottom_lines:]
                if self._is_boundary_candidate(line)
            }
            top_counts.update(top_keys)
            bottom_counts.update(bottom_keys)

        return top_counts, bottom_counts

    def _clean_page(
        self,
        text: str,
        header_keys: set[str],
        footer_keys: set[str],
    ) -> tuple[str, int]:
        lines = text.splitlines(keepends=True)
        nonempty_indices = [
            index for index, line in enumerate(lines) if line.strip()
        ]
        if not nonempty_indices:
            return text, 0

        top_indices = set(nonempty_indices[:self.top_lines])
        bottom_indices = set(nonempty_indices[-self.bottom_lines:])
        boundary_indices = top_indices | bottom_indices

        kept: list[str] = []
        removed_page_numbers = 0

        for index, line in enumerate(lines):
            candidate = self._normalize(line)
            in_boundary = index in boundary_indices

            if (
                in_boundary
                and self.remove_page_numbers
                and self._is_page_number(line)
            ):
                removed_page_numbers += 1
                continue

            if index in top_indices and candidate in header_keys:
                continue

            if index in bottom_indices and candidate in footer_keys:
                continue

            kept.append(line)

        return "".join(kept), removed_page_numbers

    @staticmethod
    def _normalize(line: str) -> str:
        return re.sub(r"\s+", " ", line.strip())

    def _is_boundary_candidate(self, line: str) -> bool:
        text = self._normalize(line)
        if not self.min_chars <= len(text) <= self.max_chars:
            return False
        if re.match(r"^(#{1,6}\s|[-*+]\s|\d+[.)、]\s*|>\s?|\|)", text):
            return False
        if any(char in text for char in "√□☑☐"):
            return False
        if self._is_page_number(text):
            return False
        return bool(re.search(r"[A-Za-z\u4e00-\u9fff]", text))

    def _is_page_number(self, line: str) -> bool:
        text = self._normalize(line)
        return any(
            pattern.fullmatch(text)
            for pattern in self.PAGE_NUMBER_PATTERNS
        )

    @staticmethod
    def _validate_positive_int(name: str, value: int) -> None:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} 必须是整数")
        if value <= 0:
            raise ValueError(f"{name} 必须大于 0")
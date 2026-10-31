import math
import re
from collections import Counter, defaultdict
from collections.abc import Iterator, Sequence
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
    """基于 PDF 页面边界识别并删除重复页眉、页脚和页码。

    识别有两条通道，取并集：

    1. **全局阈值**：同一行出现在超过 ``min_ratio`` 比例的页面顶部/底部。
    2. **连续页段**：同一行在连续 ``min_run`` 页以上稳定出现在顶部/底部。
       年报常按章节分册，前半本页眉是「公司名 + 年度报告」，后半本
       财务报表换成「截至某日止年度财务报表」，两套页眉各自占比都
       不到 60%，只靠全局阈值会一个都识别不出来（成都银行 267 页里
       有 121 处页眉残留）。分段识别就是为这种排版准备的。
    """

    PAGE_NUMBER_PATTERNS = (
        re.compile(r"\d+\s*[/／]\s*\d+"),
        re.compile(r"第\s*\d+\s*页(?:\s*[/／]\s*共\s*\d+\s*页)?"),
        re.compile(r"Page\s+\d+(?:\s+of\s+\d+)?", re.IGNORECASE),
        re.compile(r"P\.?\s*\d+", re.IGNORECASE),
        re.compile(r"[-—–]\s*\d+\s*[-—–]"),
    )

    # 页眉与页码被 PDF 抽成同一行时的行尾数字，例如 `...2025年年度报告59`
    PAGE_NUMBER_SUFFIX = re.compile(r"\s{0,2}\d{1,4}$")

    def __init__(
        self,
        *,
        top_lines: int = 2,
        bottom_lines: int = 2,
        min_pages: int = 5,
        min_count: int = 5,
        min_ratio: float = 0.6,
        min_run: int = 8,
        min_chars: int = 8,
        max_chars: int = 80,
        remove_page_numbers: bool = True,
    ) -> None:
        self._validate_positive_int("top_lines", top_lines)
        self._validate_positive_int("bottom_lines", bottom_lines)
        self._validate_positive_int("min_pages", min_pages)
        self._validate_positive_int("min_count", min_count)
        self._validate_positive_int("min_run", min_run)
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
        self.min_run = min_run
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

        top_keys_by_page, bottom_keys_by_page = self._collect_boundary_keys(pages)
        header_by_page, header_keys = self._detect_boundary_keys(top_keys_by_page)
        footer_by_page, footer_keys = self._detect_boundary_keys(bottom_keys_by_page)

        cleaned_pages: list[ParsedPage] = []
        removed_page_numbers = 0

        for index, page in enumerate(pages):
            page_headers = header_by_page[index]
            page_footers = footer_by_page[index]
            cleaned_text, removed_count = self._clean_page(
                page.text,
                page_headers,
                page_footers,
                # 页面已确认存在页眉/页脚时，边界上的裸数字按页码处理。
                strip_bare_numbers=bool(page_headers or page_footers),
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

    def _collect_boundary_keys(
        self,
        pages: Sequence[ParsedPage],
    ) -> tuple[list[set[str]], list[set[str]]]:
        """逐页收集顶部/底部的候选行 key，供后续全局与分段识别。"""
        top_keys_by_page: list[set[str]] = []
        bottom_keys_by_page: list[set[str]] = []

        for page in pages:
            lines = [
                line.strip()
                for line in page.text.splitlines()
                if line.strip()
            ]
            # 过短的页面不参与重复页眉页脚推断，避免顶部和底部区域重叠。
            if len(lines) <= self.top_lines + self.bottom_lines:
                top_keys_by_page.append(set())
                bottom_keys_by_page.append(set())
                continue

            top_keys_by_page.append(
                {
                    self._normalize(line)
                    for line in lines[:self.top_lines]
                    if self._is_boundary_candidate(line)
                }
            )
            bottom_keys_by_page.append(
                {
                    self._normalize(line)
                    for line in lines[-self.bottom_lines:]
                    if self._is_boundary_candidate(line)
                }
            )

        return top_keys_by_page, bottom_keys_by_page

    def _detect_boundary_keys(
        self,
        keys_by_page: Sequence[set[str]],
    ) -> tuple[list[set[str]], set[str]]:
        """返回「每页应删除的 key」与「被判定为页眉/页脚的全部 key」。"""
        page_count = len(keys_by_page)
        applied: list[set[str]] = [set() for _ in range(page_count)]
        if page_count < self.min_pages:
            return applied, set()

        counts: Counter[str] = Counter()
        for keys in keys_by_page:
            counts.update(keys)

        threshold = max(self.min_count, math.ceil(page_count * self.min_ratio))
        global_keys = {key for key, count in counts.items() if count >= threshold}
        for keys in applied:
            keys.update(global_keys)

        # 分段识别：不满足全局阈值的行，只要在某段连续页面上稳定出现，
        # 同样视为页眉/页脚，但只在它出现的那段页面里删除。
        positions: dict[str, list[int]] = defaultdict(list)
        for index, keys in enumerate(keys_by_page):
            for key in keys:
                positions[key].append(index)

        local_keys: set[str] = set()
        for key, indexes in positions.items():
            if key in global_keys or len(indexes) < self.min_run:
                continue
            for run in self._consecutive_runs(indexes):
                if len(run) < self.min_run:
                    continue
                local_keys.add(key)
                for index in run:
                    applied[index].add(key)

        return applied, global_keys | local_keys

    @staticmethod
    def _consecutive_runs(indexes: Sequence[int]) -> Iterator[list[int]]:
        """把升序页号切成连续段，允许中间漏掉一页（该页可能是图片页）。"""
        run: list[int] = []
        previous: int | None = None
        for index in indexes:
            if previous is not None and index - previous > 2:
                yield run
                run = []
            run.append(index)
            previous = index
        if run:
            yield run

    def _clean_page(
        self,
        text: str,
        header_keys: set[str],
        footer_keys: set[str],
        strip_bare_numbers: bool = False,
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

            if in_boundary and self.remove_page_numbers and (
                self._is_page_number(candidate)
                or (strip_bare_numbers and candidate.isdigit() and len(candidate) <= 4)
            ):
                removed_page_numbers += 1
                continue

            key = self._normalize(line)
            if index in top_indices and self._matches_header(key, header_keys):
                continue

            if index in bottom_indices and self._matches_header(key, footer_keys):
                continue

            kept.append(line)

        return "".join(kept), removed_page_numbers

    @staticmethod
    def _normalize(line: str) -> str:
        return re.sub(r"\s+", " ", line.strip())

    def _matches_header(self, key: str, keys: set[str]) -> bool:
        """先比对整行，再比对剥掉行尾页码的版本。

        ``成都银行股份有限公司 2025年年度报告59`` 与已识别的页眉
        ``成都银行股份有限公司 2025年年度报告`` 应当被视为同一行。
        只有“整行未命中、但去掉行尾数字后命中已识别页眉”时才剥，
        避免把正文里以数字结尾的普通行误删。
        """
        if key in keys:
            return True
        stripped = self.PAGE_NUMBER_SUFFIX.sub("", key).strip()
        return bool(stripped) and stripped != key and stripped in keys

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
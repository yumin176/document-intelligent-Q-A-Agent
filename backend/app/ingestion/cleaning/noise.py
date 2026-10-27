import re
from typing import Iterable, List, Optional, Pattern, Sequence, Union

from app.ingestion.cleaning.base import BaseCleaner
from app.ingestion.cleaning.code_regions import split_code_regions

_PATTERN_TYPE = type(re.compile(""))
PatternLike = Union[str, Pattern[str]]

class NoiseCleaner(BaseCleaner):
    name = "noise"

    # ---- 内置页码规则（只做整行匹配，绝不做子串删除）----
    PAGE_NUMBER_PATTERNS: Sequence[Pattern[str]] = (
        re.compile(r"第\s*\d+\s*页"),                            # 第 3 页
        re.compile(r"第\s*\d+\s*页\s*[/／]\s*共\s*\d+\s*页"),     # 第 3 页 / 共 10 页
        re.compile(r"Page\s+\d+\s+of\s+\d+", re.IGNORECASE),     # Page 3 of 10
        re.compile(r"Page\s+\d+", re.IGNORECASE),                # Page 3
        re.compile(r"P\.?\s*\d+", re.IGNORECASE),                # P. 3 / P3
        re.compile(r"[-—–]\s*\d+\s*[-—–]"),                      # - 3 -
        re.compile(r"\d+\s*[/／]\s*\d+"),                        # 15 / 26
    )

    # ---- 内置水印规则 ----
    WATERMARK_PATTERNS: Sequence[Pattern[str]] = (
        re.compile(r"机密"),                                     # 独立水印行“机密”
        re.compile(r"内部资料"),                                 # 独立水印行“内部资料”
        re.compile(r"仅供内部使用"),                             # 独立水印行“仅供内部使用”
        re.compile(r"CONFIDENTIAL", re.IGNORECASE),             # 英文水印行
    )

    def __init__(
        self,
        *,
        remove_page_numbers: bool = True,
        remove_watermarks: bool = True,
        extra_patterns: Optional[Iterable[PatternLike]] = None,
    ) -> None:
        self.remove_page_numbers=remove_page_numbers
        self.remove_watermarks=remove_watermarks
        if isinstance(extra_patterns, str):
            raise TypeError("extra_patterns 必须是正则列表，不能是单个字符串")
        try:
            # 在构造阶段编译并校验规则，避免直到清洗文档时才因非法正则失败。
            self.extra_patterns: List[Pattern[str]] = [
                p if isinstance(p, _PATTERN_TYPE) else re.compile(p)
                for p in (extra_patterns or [])
            ]
        except (re.error, TypeError) as exc:
            raise ValueError(f"extra_patterns 中存在无效正则: {exc}") from exc
        self._patterns = self._build_patterns()
    def _build_patterns(self) -> List[Pattern[str]]:
        patterns: List[Pattern[str]] = []
        if self.remove_page_numbers:
            patterns.extend(self.PAGE_NUMBER_PATTERNS)
        if self.remove_watermarks:
            patterns.extend(self.WATERMARK_PATTERNS)
        patterns.extend(self.extra_patterns)
        return patterns
    def clean(self, text: str) -> str:
        parts = split_code_regions(text)
        rendered_parts: list[str] = []
        protected: dict[str, str] = {}
        # 代码保护
        for index, item in enumerate(parts):
            if item["type"] not in {"fenced_code", "inline_code"}:
                rendered_parts.append(item["content"])
                continue

            token = f"\x00CODEX_{index}\x00"
            while token in text:
                token += "_"
            protected[token] = item["content"]
            rendered_parts.append(token)

        kept = (ln for ln in "".join(rendered_parts).splitlines(keepends=True) if not self.is_noise(ln))
        cleaned="".join(kept)
        for token, content in protected.items():
            cleaned = cleaned.replace(token, content)
        return cleaned

    def is_noise(self, line: str) -> bool:
        """判断单行是否为噪声行。用 fullmatch 保证“整行匹配”。"""
        text = line.strip()          # 仅用于匹配判断，允许行首尾空白
        if not text:
            return False             # 空行不当作噪声
        return any(p.fullmatch(text) for p in self._patterns)

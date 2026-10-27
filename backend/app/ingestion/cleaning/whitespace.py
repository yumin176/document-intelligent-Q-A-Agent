import re

from app.ingestion.cleaning.base import BaseCleaner
from app.ingestion.cleaning.code_regions import split_code_regions


class WhitespaceCleaner(BaseCleaner):
    """压缩代码区外的空白，同时保留代码内容和行首缩进。"""

    name = "whitespace"

    def __init__(self, *, max_blank_lines: int = 1) -> None:
        if isinstance(max_blank_lines, bool) or not isinstance(max_blank_lines, int):
            raise TypeError("max_blank_lines 必须是整数")
        if max_blank_lines < 0:
            raise ValueError("max_blank_lines 不能小于 0")
        self.max_blank_lines = max_blank_lines

    def clean(self, text: str) -> str:
        parts = split_code_regions(text)
        rendered_parts: list[str] = []
        protected: dict[str, str] = {}

        for index, item in enumerate(parts):
            if item["type"] not in {"fenced_code", "inline_code"}:
                rendered_parts.append(item["content"])
                continue

            token = f"\x00CODEX_{index}\x00"
            while token in text:
                token += "_"
            protected[token] = item["content"]
            rendered_parts.append(token)

        cleaned = self._clean_text("".join(rendered_parts))
        for token, content in protected.items():
            cleaned = cleaned.replace(token, content)
        return cleaned

    def _clean_text(self, text: str) -> str:
        cleaned_lines: list[str] = []
        # 按 \n 来划分行
        for line in text.split("\n"):
            # 去行尾空格
            line = line.rstrip(" \t")
            # 保留行首空格
            leading = re.match(r"^[ \t]*", line).group(0)
            body = line[len(leading):]
            # 去连续空格
            body = re.sub(r"[ \t]{2,}", " ", body)
            cleaned_lines.append(leading + body)

        cleaned = "\n".join(cleaned_lines)
        max_newlines = self.max_blank_lines + 1
        return re.sub(
            r"\n(?:[ \t]*\n)+",
            lambda match: "\n" * min(match.group(0).count("\n"), max_newlines),
            cleaned,
        )

from app.ingestion.cleaning.base import BaseCleaner
from app.ingestion.cleaning.code_regions import split_code_regions


class NormalizeCleaner(BaseCleaner):
    name = "normalize"

    def __init__(
        self,
        *,
        normalize_fullwidth: bool = True,
        remove_bom: bool = True,
        normalize_newlines: bool = True,
        remove_zero_width: bool = True,
        preserve_code: bool = True,
    ) -> None:
        self.normalize_fullwidth = normalize_fullwidth
        self.remove_bom = remove_bom
        self.normalize_newlines = normalize_newlines
        self.remove_zero_width = remove_zero_width
        self.preserve_code = preserve_code

    def clean(self, text: str) -> str:
        if self.remove_bom:
            text = text.replace("\ufeff", "")
        if self.normalize_newlines:
            text = text.replace("\r\n", "\n").replace("\r", "\n")

        parts = split_code_regions(text)
        cleaned_parts: list[str] = []

        for item in parts:
            is_code = item["type"] in {"fenced_code", "inline_code"}
            if is_code and self.preserve_code:
                cleaned_parts.append(item["content"])
            else:
                cleaned_parts.append(self._normalize_chars(item["content"]))

        return "".join(cleaned_parts)

    def _normalize_chars(self, text: str) -> str:
        if self.remove_zero_width:
            text = text.replace("\u200b", "")

        if self.normalize_fullwidth:
            chars: list[str] = []
            for ch in text:
                code = ord(ch)
                if 0xFF01 <= code <= 0xFF5E:
                    chars.append(chr(code - 0xFEE0))
                elif code == 0x3000:
                    chars.append(" ")
                else:
                    chars.append(ch)
            text = "".join(chars)

        return text

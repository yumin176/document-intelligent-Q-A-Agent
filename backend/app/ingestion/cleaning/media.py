import re

from app.ingestion.cleaning.base import BaseCleaner
from app.ingestion.cleaning.code_regions import split_code_regions

"""
| 参数 | 含义 |
|---|---|
| `keep_alt_text` | 是否把 Markdown 图片的 alt 文本保留为占位符 |
| `image_mode` | `placeholder` 替换为占位符，`drop` 直接删除 |
| `remove_base64` | 是否清理长 base64 字符串 |
| `min_base64_length` | 至少多长的 base64 才处理，默认 256 |
| `base64_placeholder` | base64 被替换后的占位文本 |
"""
class MediaCleaner(BaseCleaner):
     # ---- Markdown 图片：限制单行；URL 允许一层括号平衡 ----
    _MD_IMAGE_RE = re.compile(
        r'(?<!\\)!\[(?P<alt>[^\]\n]*)\]'
        r'\(\s*'
        r'(?P<url>(?:[^()\s]+|\([^()]*\))+)'
        r'(?:\s+(?:"[^"\n]*"|\'[^\'\n]*\'|\([^)\n]*\)))?'
        r'\s*\)'
    )
    # ---- HTML <img ...>：属性值可含 >，标签可跨行 ----
    _HTML_IMG_RE = re.compile(
        r'<img\b(?:[^>"\']|"[^"]*"|\'[^\']*\')*/?>',
        re.IGNORECASE, #re.IGNORECASE 让匹配不区分大小写。
    )

    _HTML_ALT_RE = re.compile(
        r'\balt\s*=\s*(?:"([^"]*)"|\'([^\']*)\'|([^\s>]+))',
        re.IGNORECASE,
    )

    # ---- data URI：只匹配明确的 base64 data URI，避免把普通文本误判成媒体 ----
    _DATA_URI_RE = re.compile(
        r'data:[^,\s]+;base64,[A-Za-z0-9+/=%:!$&*_\-~#@.]*',
        re.IGNORECASE,
    )

    name = "media"

    def __init__(
        self,
        *,
        keep_alt_text: bool = True,
        image_mode: str = "placeholder",
        remove_base64: bool = True,
        min_base64_length: int = 256,
        base64_placeholder: str = "[BASE64]",
    ) -> None:
        if image_mode not in ["placeholder","drop"]:
            raise ValueError("image_mode必须是'placeholder'或'drop'")
        if isinstance(min_base64_length, bool) or not isinstance(min_base64_length, int):
            raise TypeError("min_base64_length 必须是整数")
        if min_base64_length <= 0:
            raise ValueError("min_base64_length 必须大于 0")
        if not isinstance(base64_placeholder, str):
            raise TypeError("base64_placeholder 必须是字符串")
        if not base64_placeholder:
            raise ValueError("base64_placeholder 必须是非空字符串")
        self.keep_alt_text=keep_alt_text
        self.image_mode=image_mode
        self.remove_base64=remove_base64
        self.min_base64_length=min_base64_length
        self.base64_placeholder=base64_placeholder
        # 独立 base64 token：前后不能紧邻 base64 字符，长度 >= 阈值。
        # 只有带 padding 或包含 +// 的独立串才处理，避免误删长哈希和普通 ID。
        minimum = self.min_base64_length
        self._base64_padded_re = re.compile(
            rf"(?<![A-Za-z0-9+/=])[A-Za-z0-9+/]{{{minimum},}}={{1,2}}(?![A-Za-z0-9+/=])"
        )
        self._base64_symbol_re = re.compile(
            rf"(?<![A-Za-z0-9+/=])"
            rf"(?=[A-Za-z0-9+/]{{{minimum},}}(?![A-Za-z0-9+/=]))"
            rf"[A-Za-z0-9+/]*[+/][A-Za-z0-9+/]*(?![A-Za-z0-9+/=])"
        )

    def clean(self, text: str) -> str:
        """
        1. 保护代码区域
        2. 替换 Markdown 图片
        3. 替换 HTML img 标签（可选）
        4. 替换剩余 data URI
        5. 替换独立超长 base64
        6. 恢复代码区域
        """
        parts = split_code_regions(text)
        rendered_parts: list[str] = []
        protected: dict[str, str] = {}

        # 先把代码替换成 token，避免图片和 base64 规则改动代码示例。
        for index, item in enumerate(parts):
            if item["type"] not in {"fenced_code", "inline_code"}:
                rendered_parts.append(item["content"])
                continue

            token = f"\x00CODEX_{index}\x00"
            while token in text:
                token += "_"
            protected[token] = item["content"]
            rendered_parts.append(token)

        rendered_text=''.join(rendered_parts)
        rendered_text = self._replace_md_images(rendered_text)

        rendered_text = self._replace_html_imgs(rendered_text)

        if self.remove_base64:
            # 先处理带明确 data URI 前缀的内容，再处理无前缀的保守 base64 特征。
            rendered_text = self._DATA_URI_RE.sub(lambda _: self.base64_placeholder, rendered_text)
            rendered_text = self._base64_padded_re.sub(lambda _: self.base64_placeholder, rendered_text)
            rendered_text = self._base64_symbol_re.sub(lambda _: self.base64_placeholder, rendered_text)
        cleaned=rendered_text
        # 必须恢复全部代码 token，不能在这里提前 return，否则多个代码区只能恢复第一个。
        for token, content in protected.items():
            cleaned = cleaned.replace(token, content)
        return cleaned


    # markdown图片
    def _replace_md_images(self, text: str) -> str:
        return self._MD_IMAGE_RE.sub(self._md_repl, text)

    def _md_repl(self, m: "re.Match[str]") -> str:
        if self.image_mode == "drop":
            return ""
        if self.keep_alt_text:
            alt = m.group("alt").strip()
            if alt:
                return "[图片: %s]" % alt
        return "[图片]"

    # ------------------------------------------------------------------
    # HTML <img>
    # ------------------------------------------------------------------
    def _replace_html_imgs(self, text: str) -> str:
        return self._HTML_IMG_RE.sub(self._html_repl, text)

    def _html_repl(self, m: "re.Match[str]") -> str:
        if self.image_mode == "drop":
            return ""
        if self.keep_alt_text:
            am = self._HTML_ALT_RE.search(m.group(0))
            if am:
                alt = (am.group(1) or am.group(2) or am.group(3) or "").strip()
                if alt:
                    return "[图片: %s]" % alt
        return "[图片]"

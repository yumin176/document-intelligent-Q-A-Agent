import re
from hashlib import sha256
from typing import Literal

from app.ingestion.cleaning.base import BaseCleaner
from app.ingestion.cleaning.code_regions import split_code_regions

"""
| 参数 | 含义 |
|---|---|
| `min_chars` | 少于该长度的段落不参与去重，避免误删短标题和列表项 |
| `scope="adjacent"` | 只删除相邻重复段落，默认且安全 |
| `scope="document"` | 删除整篇文档中的重复段落，必须显式开启 |
"""
scope_type=Literal["adjacent","document"]
class DedupCleaner(BaseCleaner):
    name = "dedup"

    def __init__(
        self,
        *,
        min_chars: int = 20,
        scope: scope_type = "adjacent",
    ) -> None:
        # 判断是不是数字
        if isinstance(min_chars, bool) or not isinstance(min_chars, int):
            raise TypeError("min_chars 必须是整数")
        if min_chars < 0:
            raise ValueError("min_chars 不能小于 0")
        if scope not in ["adjacent","document"]:
            raise ValueError("scope必须是'adjacent'或'document'")
        self.min_chars=min_chars
        self.scope=scope
    def clean(self, text: str) -> str:
        parts = split_code_regions(text)
        rendered_parts: list[str] = []
        protected: dict[str, str] = {}
        # 将代码区域替换成token
        for index, item in enumerate(parts):
            if item["type"] not in {"fenced_code", "inline_code"}:
                rendered_parts.append(item["content"])
                continue

            # inline code 使用内容哈希生成稳定 token，这样内容相同的两个段落可以生成相同比较键。
            # fenced code 使用位置相关 token，避免两个相同的代码块被当成重复段落删除。
            if item["type"] == "inline_code":
                inline_hash = sha256(item["code"].encode("utf-8")).hexdigest()[:12]
                token = f"\x00INLINE_{inline_hash}\x00"
            else:
                token = f"\x00CODEX_{index}\x00"
            # 判断文本里有没有和token一样的内容，如果有，就在token后添加后缀，防止替换token时替换错了
            while token in text:
                token += "_"
            protected[token] = item["content"]
            rendered_parts.append(token)


        # 按 \n\n 划分段落
        paragraphs=("".join(rendered_parts)).split('\n\n')

        key_set=set()
        paragraphs_list:list[str]=[]
        before_key=''
        for paragraph in paragraphs:
            stripped_paragraph = paragraph.strip()
            # 仅由代码占位符组成的段落默认保护，避免把纯代码示例当成普通重复段落删除。
            if stripped_paragraph.startswith("\x00") and stripped_paragraph.endswith("\x00"):
                paragraphs_list.append(paragraph)
                before_key = ""
                continue
            # 段落长度小于min_chars直接跳过去重
            if len(paragraph) < self.min_chars:
                paragraphs_list.append(paragraph)
                # 中间出现受保护的短段落时，前后两个长段落不再视为相邻。
                before_key = ""
                continue
            # Markdown 标题去重；允许标题前存在最多三个空格缩进。
            if paragraph.lstrip().startswith("#"):
                paragraphs_list.append(paragraph)
                # 标题也是相邻关系边界，避免跨标题误删重复段落。
                before_key = ""
                continue
            #  r"\s+" 匹配一个或多个空白字符，包括空格、制表符 \t、换行 \n、回车 \r 等
            # 把字符串 paragraph 中所有连续的空白字符替换成一个空格，并去掉首尾空白
            key = re.sub(r"\s+", " ", paragraph).strip()
            # 删除相邻的段落
            if self.scope=="adjacent":
                # 当前key和前一个相同，直接跳过
                if key==before_key:
                    continue
                paragraphs_list.append(paragraph)
                before_key=key
            # 删除所有重复的段落
            else:
                if key in key_set:
                    continue
                key_set.add(key)
                paragraphs_list.append(paragraph)
        cleaned="\n\n".join(paragraphs_list)
        for token, content in protected.items():
            cleaned = cleaned.replace(token, content)
        return cleaned
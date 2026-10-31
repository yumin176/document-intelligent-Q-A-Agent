"""为 chunk 生成实体上下文前缀。

问题
----
年报的"主要会计数据""研发投入"等表格在多家公司之间措辞高度同构，
而 chunk 正文里通常不含公司名，实体只出现在 query 里。
结果就是"问科大讯飞，返回极米的净利润"——检索模型再强也分不清。
实测：pdf5 语料 2410 个块里只有 27% 含本公司名，
不含公司名的答案块 BM25 排名中位数是 8，含公司名的是 3。

做法
----
给每个块加一行 `【公司(代码)｜年份｜第N页】` 或 `【文件名 › 章节】`。
这一行同时进入 BM25 与 embedding，成本只有十几个字，
却把"块里没有实体"这个结构性问题直接抹平。
"""

from __future__ import annotations

import re
from pathlib import Path

# 年报文件名 → 股票代码，用于补充 query 里可能出现的代码实体
COMPANY_CODES = {
    "科大讯飞": "002230",
    "极米科技": "688696",
    "成都银行": "601838",
    "通威股份": "600438",
    "纵横股份": "688070",
    "新易盛": "300502",
    "五粮液": "000858",
}

# `极米科技_2025年年度报告` → 公司名 + 年份
_ANNUAL_REPORT_RE = re.compile(r"^(?P<company>[^_]+)_(?P<year>\d{4})年")


def parse_doc_identity(doc_name: str) -> tuple[str, str]:
    """从文档路径解析 (公司名, 年份)；不是年报命名则返回 ("", "")。"""
    stem = Path(doc_name.replace("\\", "/")).stem
    match = _ANNUAL_REPORT_RE.match(stem)
    if not match:
        return "", ""
    return match.group("company"), match.group("year")


# 标题尾部的锚点标记 `{#basic-example}` 对检索无意义，只增加噪声
_HEADING_ANCHOR_RE = re.compile(r"\s*\{#[^}]*\}\s*$")


def nearest_heading(text: str, position: int) -> str:
    """取 position 之前最近的 Markdown 标题，用于给正文块补章节信息。"""
    if position <= 0:
        return ""
    for line in reversed(text[:position].splitlines()):
        stripped = line.strip()
        if stripped.startswith("#"):
            return _HEADING_ANCHOR_RE.sub("", stripped.lstrip("#").strip())
    return ""


def build_chunk_header(
    doc_name: str,
    *,
    heading: str = "",
    page_number: int | None = None,
    block_type: str | None = None,
    table_index: int | None = None,
) -> str:
    """组装块首上下文，例如 `【极米科技(688696)｜2025年年度报告｜第42页】`。"""
    company, year = parse_doc_identity(doc_name)
    parts: list[str] = []

    if company:
        code = COMPANY_CODES.get(company)
        parts.append(f"{company}({code})" if code else company)
        if year:
            parts.append(f"{year}年年度报告")
    else:
        parts.append(Path(doc_name.replace("\\", "/")).name)
        if heading:
            parts.append(heading)

    if page_number:
        parts.append(f"第{page_number}页")
    if block_type == "table":
        parts.append(f"表{table_index}" if table_index else "表格")

    return "【" + "｜".join(parts) + "】"

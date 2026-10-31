"""chunk 实体上下文前缀的测试。"""

from app.embedding.fake import FakeEmbedder
from app.ingestion.context import (
    build_chunk_header,
    nearest_heading,
    parse_doc_identity,
)
from app.ingestion.service import build_chunk_records

ANNUAL_REPORT = "上市公司年报/极米科技_2025年年度报告.pdf"
VUE_DOC = "技术文档-Vue3中文/guide/essentials/computed.md"


def test_parse_doc_identity_extracts_company_and_year() -> None:
    assert parse_doc_identity(ANNUAL_REPORT) == ("极米科技", "2025")
    assert parse_doc_identity(VUE_DOC) == ("", "")


def test_nearest_heading_uses_closest_previous_heading() -> None:
    text = "# 计算属性\n\n第一段\n\n## 基础示例 {#basic-example}\n\n第二段"
    assert nearest_heading(text, text.index("第二段")) == "基础示例"
    assert nearest_heading(text, 0) == ""


def test_build_chunk_header_for_annual_report_includes_code() -> None:
    header = build_chunk_header(ANNUAL_REPORT)
    assert header == "【极米科技(688696)｜2025年年度报告】"


def test_build_chunk_header_for_markdown_uses_heading() -> None:
    header = build_chunk_header(VUE_DOC, heading="基础示例")
    assert header == "【computed.md｜基础示例】"


def _records(header_context: bool) -> list:
    text = "# 计算属性\n\n" + ("计算属性是基于依赖缓存的。\n" * 60)
    return build_chunk_records(
        text=text,
        file_meta={"filename": "computed.md", "file_type": "md"},
        embedder=FakeEmbedder(dim=4),
        source="computed.md",
        chunk_size=100,
        chunk_overlap=10,
        header_context=header_context,
    )


def test_header_context_prefixes_every_chunk_but_keeps_ids() -> None:
    plain = _records(False)
    prefixed = _records(True)

    assert [record.id for record in plain] == [record.id for record in prefixed]
    assert all(record.text.startswith("【computed.md") for record in prefixed)
    assert not any(record.text.startswith("【") for record in plain)
    assert all(
        record.metadata["header_context"] is True for record in prefixed
    )

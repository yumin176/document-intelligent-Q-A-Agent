r"""真实年报 PDF 的清洗回归测试。

该测试依赖本机文档，因此默认跳过。执行方式：

    $env:RUN_REAL_DOC_TEST='1'
    .\.venv\Scripts\python.exe -m pytest tests\test_annual_report_cleaning.py -s

必须通过 ANNUAL_REPORT_PDF 指定 PDF 路径。
"""

import os
import re
from pathlib import Path

import pytest

from app.ingestion.cleaning import PdfPageNoiseCleaner
from app.ingestion.cleaning.factory import build_default_cleaner_pipeline
from app.ingestion.cleaning.report import build_cleaning_report
from app.ingestion.loader import parse_document
from app.ingestion.splitter import split_text


PDF_PATH_VALUE = os.getenv("ANNUAL_REPORT_PDF", "").strip()
PDF_PATH = Path(PDF_PATH_VALUE) if PDF_PATH_VALUE else None
RUN_REAL_DOC_TEST = os.getenv("RUN_REAL_DOC_TEST") == "1"
HEADER = "极米科技股份有限公司2025年年度报告"

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not RUN_REAL_DOC_TEST or PDF_PATH is None or not PDF_PATH.exists(),
        reason="真实年报回归测试默认跳过；设置 RUN_REAL_DOC_TEST=1 后执行",
    ),
]


@pytest.fixture(scope="module")
def annual_report() -> dict:
    parsed = parse_document(PDF_PATH)
    raw_text = "\n".join(page.text for page in parsed.pages)
    page_clean_result = PdfPageNoiseCleaner().clean_pages(parsed.pages)
    page_clean_text = page_clean_result.text
    result = build_default_cleaner_pipeline().clean(page_clean_text, capture_trace=True)
    report = build_cleaning_report(
        source=str(PDF_PATH),
        raw_text=page_clean_text,
        result=result,
        include_diff=False,
    )
    chunks = split_text(result.text)

    context = {
        "raw_text": raw_text,
        "page_clean_result": page_clean_result,
        "page_clean_text": page_clean_text,
        "clean_text": result.text,
        "metadata": parsed.metadata,
        "report": report,
        "chunks": chunks,
    }

    print("\n年报清洗摘要")
    print(f"文件: {parsed.metadata['filename']}")
    print(f"pages={len(parsed.pages)}")
    print(f"raw_chars={len(raw_text)}")
    print(f"page_clean_chars={len(page_clean_text)}")
    print(f"cleaned_chars={len(result.text)}")
    print(f"removed_page_numbers={page_clean_result.removed_page_numbers}")
    print(f"removed_header_keys={page_clean_result.removed_header_keys}")
    print(f"removed_footer_keys={page_clean_result.removed_footer_keys}")
    print(f"header_before={raw_text.count(HEADER)}")
    print(f"header_after={result.text.count(HEADER)}")
    print(f"chunks={len(chunks)}")

    return context


def test_page_numbers_and_header_are_removed(annual_report: dict) -> None:
    raw_text = annual_report["raw_text"]
    clean_text = annual_report["clean_text"]
    page_marker = re.compile(r"^\s*\d+\s*/\s*269\s*$", re.MULTILINE)

    assert len(page_marker.findall(raw_text)) == 269
    assert page_marker.findall(clean_text) == []
    assert raw_text.count(HEADER) == 258
    assert clean_text.count(HEADER) == 0
    assert len(clean_text) < len(raw_text)


def test_pdf_page_cleaner_summary(annual_report: dict) -> None:
    result = annual_report["page_clean_result"]

    assert len(result.pages) == 269
    assert result.removed_page_numbers == 269
    assert result.removed_header_keys == (HEADER,)
    assert result.removed_footer_keys == ()
    assert HEADER not in result.text


def test_cleaned_report_can_be_split_into_chunks(annual_report: dict) -> None:
    chunks = annual_report["chunks"]

    assert chunks
    assert all(chunk.strip() for chunk in chunks)
    assert max(len(chunk) for chunk in chunks) <= 500
    assert len(chunks) < 425


def test_cleaning_report_summary(annual_report: dict) -> None:
    report = annual_report["report"]

    assert report["raw_chars"] == len(annual_report["page_clean_text"])
    assert report["cleaned_chars"] == len(annual_report["clean_text"])
    assert [step["cleaner"] for step in report["steps"]] == [
        "normalize",
        "media",
        "noise",
        "whitespace",
        "dedup",
    ]
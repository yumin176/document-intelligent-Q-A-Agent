import hashlib
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.api.routes.documents import router as documents_router  # noqa: F401
from app.dependencies import get_bm25_refresher, get_ingestion_service
from app.embedding.fake import FakeEmbedder
from app.ingestion.cleaning import CleanerPipeline, PdfPageNoiseCleaner
from app.ingestion.cleaning.factory import build_default_cleaner_pipeline
from app.ingestion.loader import ParsedDocument, ParsedPage
from app.ingestion.service import IngestionService, build_chunk_records
from app.main import create_app
from app.vectorstore.memory import InMemoryVectorStore


def make_service(
    cleaner: CleanerPipeline | None = None,
    page_cleaner: PdfPageNoiseCleaner | None = None,
) -> tuple[IngestionService, InMemoryVectorStore]:
    store = InMemoryVectorStore()
    service = IngestionService(
        embedder=FakeEmbedder(dim=8),
        vector_store=store,
        cleaner=cleaner,
        page_cleaner=page_cleaner,
    )
    return service, store


def test_default_cleaner_pipeline_order() -> None:
    pipeline = build_default_cleaner_pipeline()

    assert [cleaner.name for cleaner in pipeline.cleaners] == [
        "normalize",
        "media",
        "noise",
        "whitespace",
        "dedup",
    ]


def test_ingest_file_uses_cleaned_text(tmp_path: Path) -> None:
    raw = (
        "\ufeffＡＰＩ   文档\r\n\r\n"
        "第 1 页\r\n\r\n"
        "这是需要在清洗之后保留到知识库中的完整段落。\r\n\r\n"
        "这是需要在清洗之后保留到知识库中的完整段落。\r\n\r\n"
        "![架构图](data:image/png;base64,AAAA)"
    )
    repeated = "这是需要在清洗之后保留到知识库中的完整段落。"
    expected = f"API 文档\n\n{repeated}\n\n[图片: 架构图]"
    path = tmp_path / "noisy.md"
    path.write_text(raw, encoding="utf-8")
    service, store = make_service()

    result = service.ingest_file(path)
    chunks = store.all_chunks()

    assert result.chunk_count == 1
    assert len(chunks) == 1
    assert chunks[0].text == expected
    assert chunks[0].text.count(repeated) == 1
    assert "\ufeff" not in chunks[0].text
    assert "第 1 页" not in chunks[0].text
    assert "data:image" not in chunks[0].text

    metadata = chunks[0].metadata
    assert metadata["raw_num_chars"] == len(raw)
    assert metadata["num_chars"] == len(expected)
    assert metadata["cleaning_steps"] == [
        "normalize",
        "media",
        "noise",
        "whitespace",
        "dedup",
    ]
    assert metadata["content_hash"] == hashlib.sha256(
        expected.encode("utf-8")
    ).hexdigest()
    assert metadata["document_id"] == result.document_id
    assert metadata["page_cleaning"] == {"enabled": False}


def test_ingest_file_can_inject_custom_pipeline(tmp_path: Path) -> None:
    raw = "Ａ   文本"
    path = tmp_path / "raw.md"
    path.write_text(raw, encoding="utf-8")
    service, store = make_service(cleaner=CleanerPipeline([]))

    service.ingest_file(path)
    chunk = store.all_chunks()[0]

    assert chunk.text == raw
    assert chunk.metadata["cleaning_steps"] == []
    assert chunk.metadata["raw_num_chars"] == len(raw)
    assert chunk.metadata["num_chars"] == len(raw)


def test_empty_cleaned_document_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "only-watermark.md"
    path.write_text("机密\n", encoding="utf-8")
    service, store = make_service()

    with pytest.raises(ValueError, match="清洗后文档为空"):
        service.ingest_file(path)

    assert store.all_chunks() == []


def test_upload_returns_400_when_cleaning_produces_empty_text() -> None:
    app = create_app()
    refresh_calls: list[bool] = []

    def service_override() -> IngestionService:
        service, _ = make_service()
        return service

    def refresher_override():
        return lambda: refresh_calls.append(True)

    app.dependency_overrides[get_ingestion_service] = service_override
    app.dependency_overrides[get_bm25_refresher] = refresher_override
    client = TestClient(app)

    files = {"file": ("empty.md", "机密".encode("utf-8"), "text/markdown")}
    response = client.post("/api/documents", files=files)

    assert response.status_code == 400
    assert "清洗后文档为空" in response.json()["detail"]
    assert refresh_calls == []


def test_build_chunk_records_supports_metadata_without_clean_summary() -> None:
    file_meta = {"filename": "sample.md", "file_type": "md"}
    records = build_chunk_records(
        text="hello",
        file_meta=file_meta,
        embedder=FakeEmbedder(dim=8),
        source="sample.md",
    )

    assert len(records) == 1
    assert records[0].metadata["raw_num_chars"] == len("hello")
    assert records[0].metadata["num_chars"] == len("hello")
    assert records[0].metadata["cleaning_steps"] == []
    assert records[0].metadata["page_cleaning"] == {}
class SpyPageCleaner:
    def __init__(self) -> None:
        self.calls = 0

    def clean_pages(self, pages):
        self.calls += 1
        raise AssertionError("非 PDF 文件不应调用页级清洗")


def test_non_pdf_document_skips_page_cleaner(tmp_path: Path) -> None:
    spy = SpyPageCleaner()
    path = tmp_path / "sample.md"
    path.write_text("正文", encoding="utf-8")
    service, store = make_service(page_cleaner=spy)  # type: ignore[arg-type]

    service.ingest_file(path)

    assert spy.calls == 0
    assert store.all_chunks()[0].metadata["page_cleaning"] == {"enabled": False}


def test_pdf_document_uses_page_cleaner_and_records_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    header = "某某科技股份有限公司2025年年度报告"
    footer = "公司页脚仅供内部使用说明"
    pages = tuple(
        ParsedPage(
            page_number=index,
            text=(
                f"{header}\n"
                f"{index} / 10\n"
                f"正文第{index}页\n"
                "补充内容 A\n"
                f"补充内容 B{index}\n"
                f"{footer}"
            ),
        )
        for index in range(1, 11)
    )
    parsed = ParsedDocument(
        pages=pages,
        metadata={
            "filename": "report.pdf",
            "file_type": "pdf",
            "file_size": 123,
            "page_count": len(pages),
        },
    )
    path = tmp_path / "report.pdf"
    path.write_bytes(b"fake pdf")
    monkeypatch.setattr("app.ingestion.service.parse_document", lambda _: parsed)
    service, store = make_service()

    result = service.ingest_file(path)
    chunk = store.all_chunks()[0]
    metadata = chunk.metadata

    assert result.chunk_count == 1
    assert header not in chunk.text
    assert footer not in chunk.text
    assert "/ 10" not in chunk.text
    assert metadata["cleaning_steps"][0] == "pdf_page_noise"
    assert metadata["raw_num_chars"] == len(parsed.text)
    assert metadata["num_chars"] == len(chunk.text)
    assert metadata["page_cleaning"]["enabled"] is True
    assert metadata["page_cleaning"]["removed_page_numbers"] == 10
    assert metadata["page_cleaning"]["removed_header_keys"] == [header]
    assert metadata["page_cleaning"]["removed_footer_keys"] == [footer]
    assert metadata["page_cleaning"]["before_chars"] == len(parsed.text)
    assert metadata["page_cleaning"]["after_chars"] < len(parsed.text)
from types import SimpleNamespace

from app.schemas import SourceOut


def test_source_out_uses_metadata_locator_fields() -> None:
    source = SimpleNamespace(
        chunk_id="doc1#3:abcdef12",
        score=0.5,
        text="块内容",
        metadata={
            "anchor_id": "doc1#3:abcdef12",
            "document_id": "doc1",
            "filename": "a.md",
            "chunk_index": 3,
            "page_number": 7,
        },
    )

    out = SourceOut.from_source(source)

    assert out.anchor_id == "doc1#3:abcdef12"
    assert out.document_id == "doc1"
    assert out.filename == "a.md"
    assert out.chunk_index == 3
    assert out.page_number == 7


def test_source_out_falls_back_to_chunk_id() -> None:
    source = SimpleNamespace(chunk_id="doc#0", score=0.1, text="x", metadata={})

    out = SourceOut.from_source(source)

    assert out.anchor_id == "doc#0"
    assert out.document_id == ""
    assert out.filename == ""
    assert out.chunk_index == 0
    assert out.page_number == 0

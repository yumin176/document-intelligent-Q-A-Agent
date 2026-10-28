import re
from hashlib import sha1, sha256

from app.embedding.fake import FakeEmbedder
from app.ingestion.service import build_anchor_id, build_chunk_records


ANCHOR_PATTERN = re.compile(r"^[0-9a-f]{32}#\d+:[0-9a-f]{8}$")


def make_records(
    text: str,
    *,
    filename: str = "sample.md",
    chunk_strategy: str = "fixed",
    chunk_size: int = 500,
    chunk_overlap: int = 50,
):
    return build_chunk_records(
        text=text,
        file_meta={"filename": filename, "file_type": "md"},
        embedder=FakeEmbedder(dim=8),
        source=filename,
        chunk_strategy=chunk_strategy,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )


def test_anchor_ids_are_stable_for_same_content() -> None:
    text = "这是一段用于验证锚点稳定性的正文。" * 40

    first = make_records(text)
    second = make_records(text)

    assert [record.id for record in first] == [record.id for record in second]


def test_anchor_metadata_is_complete() -> None:
    record = make_records("用于验证锚点元数据的正文。")[0]
    metadata = record.metadata

    assert record.id == metadata["anchor_id"]
    assert metadata["anchor_id"] == build_anchor_id(
        metadata["document_id"],
        metadata["chunk_index"],
        record.text,
    )
    assert metadata["doc_name"] == "sample.md"
    assert metadata["chunk_hash"] == sha256(record.text.encode("utf-8")).hexdigest()
    assert ANCHOR_PATTERN.fullmatch(record.id)
    assert "/" not in record.id
    assert "\\" not in record.id
    assert " " not in record.id


def test_anchor_prefix_hash_matches_contract() -> None:
    record = make_records("abcdefghij" * 10)[0]
    expected_prefix = sha1(record.text[:50].encode("utf-8")).hexdigest()[:8]

    assert record.id.endswith(f":{expected_prefix}")


def test_different_filename_changes_document_and_anchor_ids() -> None:
    first = make_records("相同正文内容。", filename="one.md")[0]
    second = make_records("相同正文内容。", filename="two.md")[0]

    assert first.metadata["document_id"] != second.metadata["document_id"]
    assert first.id != second.id


def test_changed_prefix_changes_anchor_id() -> None:
    first = make_records("A" * 100)[0]
    second = make_records("B" * 100)[0]

    assert first.id != second.id
    assert first.metadata["chunk_hash"] != second.metadata["chunk_hash"]


def test_suffix_change_is_detected_by_chunk_hash_even_if_anchor_is_stable() -> None:
    first = make_records("A" * 50 + "B" * 50)[0]
    second = make_records("A" * 50 + "C" * 50)[0]

    assert first.id == second.id
    assert first.metadata["chunk_hash"] != second.metadata["chunk_hash"]


def test_different_chunk_indexes_have_different_anchor_ids() -> None:
    records = make_records("A" * 1200, chunk_size=500, chunk_overlap=0)

    assert len(records) >= 2
    assert records[0].id != records[1].id
    assert records[0].metadata["chunk_index"] == 0
    assert records[1].metadata["chunk_index"] == 1


def test_fixed_and_recursive_chunks_have_valid_anchors() -> None:
    text = "".join(
        f"第{index}句包含一些用于锚点测试的正文内容。"
        for index in range(30)
    )

    for strategy in ("fixed", "recursive"):
        records = make_records(
            text,
            chunk_strategy=strategy,
            chunk_size=60,
            chunk_overlap=10,
        )
        assert records
        assert all(ANCHOR_PATTERN.fullmatch(record.id) for record in records)
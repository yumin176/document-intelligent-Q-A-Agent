from app.ingestion.pdf_table_aware import (
    PdfBlock,
    split_markdown_table,
    split_table_aware_blocks,
    table_to_markdown,
)


def test_table_to_markdown_normalizes_cells() -> None:
    markdown = table_to_markdown(
        [
            ["项目", "2025年", "备注"],
            ["营业收入", "3,467,484,106.01", "第一行\n第二行"],
            [None, "1|2", ""],
        ],
        page_number=9,
        table_index=2,
    )

    assert markdown.startswith("[表格 第9页 表2]")
    assert "| 项目 | 2025年 | 备注 |" in markdown
    assert "| 营业收入 | 3,467,484,106.01 | 第一行 第二行 |" in markdown
    assert "1\\|2" in markdown


def test_small_table_stays_atomic() -> None:
    markdown = table_to_markdown(
        [["项目", "值"], ["收入", "100"]],
        page_number=1,
        table_index=1,
    )

    chunks = split_markdown_table(
        markdown,
        chunk_size=500,
        metadata={"page_number": 1, "table_id": "p1-t1"},
    )

    assert len(chunks) == 1
    assert chunks[0].text == markdown
    assert chunks[0].block_type == "table"


def test_large_table_splits_and_repeats_header() -> None:
    rows = [["项目", "2025年"]]
    rows.extend([[f"项目{index}", str(index)] for index in range(20)])
    markdown = table_to_markdown(rows, page_number=2, table_index=3)

    chunks = split_markdown_table(
        markdown,
        chunk_size=180,
        metadata={"page_number": 2, "table_id": "p2-t3"},
    )

    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.text.startswith("[表格 第2页 表3]")
        assert "| 项目 | 2025年 |" in chunk.text
        assert len(chunk.text) <= 180
    assert chunks[0].metadata["row_start"] == 1
    assert chunks[-1].metadata["row_end"] == 20


def test_split_blocks_keeps_text_and_table_types() -> None:
    blocks = (
        PdfBlock(block_type="text", text="正文" * 300, page_number=1),
        PdfBlock(
            block_type="table",
            text=table_to_markdown(
                [["项目", "值"], ["收入", "100"]],
                page_number=1,
                table_index=1,
            ),
            page_number=1,
            metadata={"table_id": "p1-t1"},
        ),
    )

    chunks = split_table_aware_blocks(blocks, chunk_size=100, chunk_overlap=10)

    assert any(chunk.block_type == "text" for chunk in chunks)
    assert any(chunk.block_type == "table" for chunk in chunks)
    assert all(len(chunk.text) <= 100 for chunk in chunks if chunk.block_type == "text")
def test_table_aware_text_strategy_is_passed(monkeypatch) -> None:
    calls = []

    def fake_split_text(text, *, chunk_size, chunk_overlap, strategy):
        calls.append(strategy)
        return [text]

    monkeypatch.setattr("app.ingestion.pdf_table_aware.split_text", fake_split_text)
    blocks = (PdfBlock(block_type="text", text="正文", page_number=1),)

    split_table_aware_blocks(blocks, strategy="recursive")

    assert calls == ["recursive"]
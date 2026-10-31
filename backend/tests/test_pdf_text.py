from app.ingestion.loader import ParsedDocument, ParsedPage
from app.ingestion.pdf_text import dewrap_page_text, dewrap_pdf_pages, join_pdf_pages


def test_chinese_visual_wraps_are_merged_without_spaces() -> None:
    raw = "这是一段因为 PDF 排版被拆开的\n中文正文。"

    assert dewrap_page_text(raw) == "这是一段因为 PDF 排版被拆开的中文正文。"


def test_english_visual_wraps_are_merged_with_one_space() -> None:
    raw = "This is a wrapped\nEnglish sentence."

    assert dewrap_page_text(raw) == "This is a wrapped English sentence."


def test_sentence_end_starts_new_paragraph() -> None:
    raw = "第一句已经结束。\n第二句是新段落。"

    assert dewrap_page_text(raw) == "第一句已经结束。\n\n第二句是新段落。"


def test_heading_and_list_start_new_paragraphs() -> None:
    raw = "标题\n（一）第一项\n- 列表内容\n| 表格行 |"

    result = dewrap_page_text(raw)

    assert result == "标题\n\n（一）第一项\n\n- 列表内容\n\n| 表格行 |"


def test_blank_lines_are_removed_between_rebuilt_paragraphs() -> None:
    raw = "第一段第一行\n第一段第二行\n\n第二段"

    assert dewrap_page_text(raw) == "第一段第一行第一段第二行\n\n第二段"


def test_pdf_pages_are_joined_with_double_newline() -> None:
    pages = (
        ParsedPage(page_number=1, text="第一页\n"),
        ParsedPage(page_number=2, text="第二页\n"),
    )

    assert join_pdf_pages(dewrap_pdf_pages(pages)) == "第一页\n\n第二页"


def test_parsed_document_text_uses_page_boundaries() -> None:
    document = ParsedDocument(
        pages=(
            ParsedPage(page_number=1, text="第一页"),
            ParsedPage(page_number=2, text="第二页"),
        ),
        metadata={},
    )

    assert document.text == "第一页\n\n第二页"
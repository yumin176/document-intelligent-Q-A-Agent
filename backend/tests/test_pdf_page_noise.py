import pytest

from app.ingestion.cleaning import PageCleanResult, PdfPageNoiseCleaner
from app.ingestion.loader import ParsedPage


HEADER = "某某科技股份有限公司2025年年度报告"
FOOTER = "公司页脚仅供内部使用说明"


def make_pages(count: int = 10, include_page_numbers: bool = True) -> tuple[ParsedPage, ...]:
    pages = []
    for index in range(1, count + 1):
        marker = f"{index} / {count}\n" if include_page_numbers else ""
        pages.append(
            ParsedPage(
                page_number=index,
                text=(
                    f"{HEADER}\n"
                    f"{marker}"
                    f"正文第{index}页内容\n"
                    "补充正文内容 A\n"
                    f"补充正文内容 B{index}\n"
                    f"{FOOTER}"
                ),
            )
        )
    return tuple(pages)


def test_removes_repeated_header_footer_and_page_numbers() -> None:
    result = PdfPageNoiseCleaner().clean_pages(make_pages())

    assert result.removed_header_keys == (HEADER,)
    assert result.removed_footer_keys == (FOOTER,)
    assert result.removed_page_numbers == 10
    assert all(HEADER not in page.text for page in result.pages)
    assert all(FOOTER not in page.text for page in result.pages)
    assert all("/ 10" not in page.text for page in result.pages)
    assert result.pages[0].text.strip() == "正文第1页内容\n补充正文内容 A\n补充正文内容 B1"


def test_repeated_header_footer_do_not_require_page_number_lines() -> None:
    pages = tuple(
        ParsedPage(
            page_number=index,
            text=f"{HEADER}\n正文第{index}页\n补充内容\n更多内容\n{FOOTER}",
        )
        for index in range(1, 11)
    )

    result = PdfPageNoiseCleaner().clean_pages(pages)

    assert HEADER not in result.text
    assert FOOTER not in result.text


def test_checkbox_lines_are_not_removed() -> None:
    pages = tuple(
        ParsedPage(
            page_number=index,
            text=f"□适用 √不适用\n{index} / 10\n正文第{index}页",
        )
        for index in range(1, 11)
    )

    result = PdfPageNoiseCleaner().clean_pages(pages)

    assert all("□适用 √不适用" in page.text for page in result.pages)
    assert result.removed_page_numbers == 10


def test_repeated_body_line_in_middle_is_preserved() -> None:
    repeated = "中间重复正文内容用于测试"
    pages = tuple(
        ParsedPage(
            page_number=index,
            text=(
                f"{HEADER}\n"
                f"{index} / 10\n"
                f"{repeated}\n"
                "正文内容 A\n"
                "正文内容 B\n"
                "正文内容 C"
            ),
        )
        for index in range(1, 11)
    )

    result = PdfPageNoiseCleaner().clean_pages(pages)

    assert all(repeated in page.text for page in result.pages)


def test_too_few_pages_does_not_remove_repeated_boundaries() -> None:
    result = PdfPageNoiseCleaner().clean_pages(make_pages(count=3))

    assert result.removed_header_keys == ()
    assert result.removed_footer_keys == ()
    assert all(HEADER in page.text for page in result.pages)
    assert all(FOOTER in page.text for page in result.pages)
    assert result.removed_page_numbers == 3


def test_header_below_ratio_threshold_is_preserved() -> None:
    pages = []
    for index in range(1, 11):
        header = HEADER if index <= 5 else "另一个不同的页眉内容"
        pages.append(
            ParsedPage(
                page_number=index,
                text=f"{header}\n正文第{index}页\n补充内容\n更多内容",
            )
        )

    result = PdfPageNoiseCleaner().clean_pages(tuple(pages))

    assert result.removed_header_keys == ()
    assert HEADER in result.text


def test_page_number_removal_can_be_disabled() -> None:
    result = PdfPageNoiseCleaner(remove_page_numbers=False).clean_pages(
        make_pages(count=3)
    )

    assert result.removed_page_numbers == 0
    assert all(" / 3" in page.text for page in result.pages)


def test_preserves_crlf_for_kept_lines() -> None:
    pages = (
        ParsedPage(
            page_number=1,
            text=f"{HEADER}\r\n1 / 1\r\n正文内容\r\n{FOOTER}",
        ),
    )

    result = PdfPageNoiseCleaner().clean_pages(pages)

    # 单页文档不触发重复页眉页脚推断，但边界页码仍应删除。
    assert result.removed_page_numbers == 1
    assert result.pages[0].text == f"{HEADER}\r\n正文内容\r\n{FOOTER}"


def test_cleaning_is_idempotent() -> None:
    cleaner = PdfPageNoiseCleaner()
    first = cleaner.clean_pages(make_pages())
    second = cleaner.clean_pages(first.pages)

    assert second.pages == first.pages


def test_empty_page_sequence() -> None:
    result = PdfPageNoiseCleaner().clean_pages([])

    assert result == PageCleanResult(pages=())
    assert result.text == ""


def test_text_property_joins_pages_with_single_newline() -> None:
    pages = (
        ParsedPage(page_number=1, text="第一页\n"),
        ParsedPage(page_number=2, text="第二页\n"),
    )

    result = PdfPageNoiseCleaner().clean_pages(pages)

    assert result.text == "第一页\n第二页"


def test_invalid_configuration_is_rejected() -> None:
    with pytest.raises(ValueError, match="min_ratio"):
        PdfPageNoiseCleaner(min_ratio=0)
    with pytest.raises(ValueError, match="min_chars"):
        PdfPageNoiseCleaner(min_chars=100, max_chars=10)
    with pytest.raises(TypeError, match="整数"):
        PdfPageNoiseCleaner(top_lines=1.5)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="布尔值"):
        PdfPageNoiseCleaner(remove_page_numbers="yes")  # type: ignore[arg-type]
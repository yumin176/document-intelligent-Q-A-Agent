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


def test_segmented_headers_are_removed_without_global_ratio() -> None:
    """年报分章节换页眉时，两套页眉各自不到 60% 也要能识别。"""
    first = "成都银行股份有限公司2025年年度报告"
    second = "截至2025年12月31日止年度财务报表"
    pages = tuple(
        ParsedPage(
            page_number=index,
            text=(
                f"{first if index <= 12 else second}\n"
                f"正文第{index}页\n"
                f"补充内容一{index}\n"
                f"补充内容二{index}\n"
                f"补充内容三{index}"
            ),
        )
        for index in range(1, 25)
    )

    result = PdfPageNoiseCleaner().clean_pages(pages)

    assert set(result.removed_header_keys) == {first, second}
    assert first not in result.text
    assert second not in result.text
    assert "补充内容一1" in result.text


def test_short_header_run_is_preserved() -> None:
    """只在少数连续页出现的行，不按页眉处理。"""
    pages = tuple(
        ParsedPage(
            page_number=index,
            text=(
                f"{HEADER if index <= 6 else '另一份文件使用的页眉内容'}\n"
                f"正文第{index}页\n补充内容一{index}\n"
                f"补充内容二{index}\n补充内容三{index}"
            ),
        )
        for index in range(1, 21)
    )

    result = PdfPageNoiseCleaner(min_run=8).clean_pages(pages)

    assert result.removed_header_keys == ("另一份文件使用的页眉内容",)
    assert HEADER in result.text


def test_header_with_glued_page_number_is_removed() -> None:
    """`...年度报告7` 这类页码粘在页眉尾部的行，要能跟随页眉一起删掉。"""
    pages = tuple(
        ParsedPage(
            page_number=index,
            text=(
                f"{HEADER if index != 7 else f'{HEADER}7'}\n"
                f"正文第{index}页\n补充内容一{index}\n"
                f"补充内容二{index}\n补充内容三{index}"
            ),
        )
        for index in range(1, 12)
    )

    result = PdfPageNoiseCleaner().clean_pages(pages)

    assert HEADER not in result.text
    assert "正文第7页" in result.text


def test_bare_page_numbers_are_removed_when_header_detected() -> None:
    """页眉已确认时，页边界的裸数字按页码删除，避免拼接进正文。"""
    pages = tuple(
        ParsedPage(
            page_number=index,
            text=(
                f"{HEADER}\n{index}\n正文第{index}页\n"
                f"补充内容一{index}\n补充内容二{index}"
            ),
        )
        for index in range(1, 11)
    )

    result = PdfPageNoiseCleaner().clean_pages(pages)

    assert result.removed_page_numbers == 10
    assert result.pages[0].text.strip().startswith("正文第1页")
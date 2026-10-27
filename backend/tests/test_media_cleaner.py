import base64

import pytest

from app.ingestion.cleaning import CleanerPipeline, MediaCleaner, NormalizeCleaner


def test_media_cleaner():
    raw = "前文 ![架构图](data:image/png;base64,AAAA...) 后文"
    expected = "前文 [图片: 架构图] 后文"
    cleaner=MediaCleaner()
    res=cleaner.clean(raw)
    assert res==expected
def test_empty_alt_uses_generic_placeholder() -> None:
    raw = "前 ![](https://example.com/a.png) 后"

    assert MediaCleaner().clean(raw) == "前 [图片] 后"


def test_markdown_image_with_title_is_replaced() -> None:
    raw = '前 ![架构图](https://example.com/a.png "标题") 后'

    assert MediaCleaner().clean(raw) == "前 [图片: 架构图] 后"


def test_alt_text_can_be_disabled() -> None:
    raw = "前 ![架构图](https://example.com/a.png) 后"

    assert MediaCleaner(keep_alt_text=False).clean(raw) == "前 [图片] 后"


def test_markdown_images_can_be_dropped() -> None:
    raw = "![架构图](https://example.com/a.png)"

    assert MediaCleaner(image_mode="drop").clean(raw) == ""


def test_multiple_markdown_images_are_replaced() -> None:
    raw = "![一](a.png) 和 ![二](b.png)"

    assert MediaCleaner().clean(raw) == "[图片: 一] 和 [图片: 二]"


def test_regular_markdown_link_is_preserved() -> None:
    raw = "[文档](https://example.com/docs)"

    assert MediaCleaner().clean(raw) == raw


def test_escaped_markdown_image_is_preserved() -> None:
    raw = r"\![架构图](https://example.com/a.png)"

    assert MediaCleaner().clean(raw) == raw


def test_html_image_alt_text_is_preserved() -> None:
    raw = '<img src="a.png" alt="架构图">'

    assert MediaCleaner().clean(raw) == "[图片: 架构图]"


def test_html_image_without_alt_uses_generic_placeholder() -> None:
    raw = '<img src="a.png">'

    assert MediaCleaner().clean(raw) == "[图片]"


def test_standalone_base64_data_uri_is_replaced() -> None:
    raw = "数据：data:text/plain;base64,SGVsbG8="

    assert MediaCleaner().clean(raw) == "数据：[BASE64]"


def test_non_base64_data_uri_is_preserved() -> None:
    raw = "data:text/plain,hello"

    assert MediaCleaner().clean(raw) == raw


def test_long_padded_base64_is_replaced() -> None:
    payload = base64.b64encode(b"x" * 301).decode("ascii")
    raw = f"前文 {payload} 后文"

    assert MediaCleaner(min_base64_length=100).clean(raw) == "前文 [BASE64] 后文"


def test_long_base64_with_symbols_is_replaced() -> None:
    payload = "A" * 20 + "/" + "B" * 20
    raw = f"前文 {payload} 后文"

    assert MediaCleaner(min_base64_length=16).clean(raw) == "前文 [BASE64] 后文"


def test_short_base64_like_string_is_preserved() -> None:
    raw = "SGVsbG8="

    assert MediaCleaner(min_base64_length=100).clean(raw) == raw


def test_long_hex_hash_is_not_treated_as_base64() -> None:
    raw = "a" * 300

    assert MediaCleaner(min_base64_length=100).clean(raw) == raw


def test_base64_cleaning_can_be_disabled() -> None:
    payload = base64.b64encode(b"x" * 301).decode("ascii")
    raw = f"data:text/plain;base64,SGVsbG8= {payload}"

    assert MediaCleaner(remove_base64=False).clean(raw) == raw


def test_all_code_regions_are_restored() -> None:
    raw = (
        "```text\n"
        "![代码内图片](code.png)\n"
        "data:text/plain;base64,SGVsbG8=\n"
        "```\n"
        "正文 ![架构图](a.png)\n"
        "`data:text/plain;base64,SGVsbG8=`"
    )
    expected = (
        "```text\n"
        "![代码内图片](code.png)\n"
        "data:text/plain;base64,SGVsbG8=\n"
        "```\n"
        "正文 [图片: 架构图]\n"
        "`data:text/plain;base64,SGVsbG8=`"
    )

    assert MediaCleaner().clean(raw) == expected


def test_invalid_image_mode_is_rejected() -> None:
    with pytest.raises(ValueError, match="image_mode"):
        MediaCleaner(image_mode="invalid")


def test_invalid_min_base64_length_is_rejected() -> None:
    with pytest.raises(ValueError, match="大于 0"):
        MediaCleaner(min_base64_length=0)

    with pytest.raises(TypeError, match="整数"):
        MediaCleaner(min_base64_length=1.5)  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="整数"):
        MediaCleaner(min_base64_length=True)  # type: ignore[arg-type]


def test_invalid_base64_placeholder_is_rejected() -> None:
    with pytest.raises(TypeError, match="字符串"):
        MediaCleaner(base64_placeholder=123)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="非空"):
        MediaCleaner(base64_placeholder="")


def test_empty_string_returns_empty_string() -> None:
    assert MediaCleaner().clean("") == ""


def test_media_cleaning_is_idempotent() -> None:
    cleaner = MediaCleaner(min_base64_length=16)
    raw = "前文 ![图](a.png) 和 data:text/plain;base64,SGVsbG8= 后文"

    cleaned_once = cleaner.clean(raw)
    cleaned_twice = cleaner.clean(cleaned_once)

    assert cleaned_twice == cleaned_once


def test_media_cleaner_works_inside_pipeline() -> None:
    pipeline = CleanerPipeline([NormalizeCleaner(), MediaCleaner()])
    raw = "前文\r\n![架构图](data:image/png;base64,AAAA)\r\n后文"

    result = pipeline.clean(raw)

    assert result.text == "前文\n[图片: 架构图]\n后文"
    assert [step.cleaner_name for step in result.steps] == ["normalize", "media"]
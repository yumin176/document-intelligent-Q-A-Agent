from app.ingestion.cleaning.code_regions import split_code_regions


def test_plain_text_is_returned_as_one_text_part() -> None:
    parts = split_code_regions("普通Ａ文本")

    assert parts == [{"type": "text", "content": "普通Ａ文本"}]


def test_empty_text_returns_no_parts() -> None:
    assert split_code_regions("") == []


def test_fenced_code_is_separated_from_surrounding_text() -> None:
    text = "前文\n```python\nprint('Ａ')\n```\n后文"

    parts = split_code_regions(text)

    assert [part["type"] for part in parts] == ["text", "fenced_code", "text"]
    assert parts[0]["content"] == "前文\n"
    assert parts[1]["content"] == "```python\nprint('Ａ')\n```\n"
    assert parts[1]["code"] == "print('Ａ')\n"
    assert parts[1]["lang"] == "python"
    assert parts[1]["closed"] is True
    assert parts[2]["content"] == "后文"


def test_inline_code_is_separated_from_text() -> None:
    parts = split_code_regions("前Ａ`codeＡ`后")

    assert [part["type"] for part in parts] == ["text", "inline_code", "text"]
    assert parts[0]["content"] == "前Ａ"
    assert parts[1]["content"] == "`codeＡ`"
    assert parts[1]["code"] == "codeＡ"
    assert parts[2]["content"] == "后"


def test_tilde_fence_is_supported() -> None:
    parts = split_code_regions("前\n~~~js\nconst Ａ = 1;\n~~~\n后")

    assert parts[1]["type"] == "fenced_code"
    assert parts[1]["lang"] == "js"
    assert parts[1]["closed"] is True
    assert parts[1]["code"] == "const Ａ = 1;\n"


def test_longer_backtick_fence_can_contain_shorter_backticks() -> None:
    text = "````md\n```js\nconst a = 1;\n```\n````"
    parts = split_code_regions(text)

    assert parts[0]["type"] == "fenced_code"
    assert parts[0]["closed"] is True
    assert parts[0]["code"] == "```js\nconst a = 1;\n```\n"


def test_unclosed_fence_is_protected_until_end_of_text() -> None:
    text = "前文\n```js\nconst Ａ = 1;"

    parts = split_code_regions(text)

    assert [part["type"] for part in parts] == ["text", "fenced_code"]
    assert parts[1]["closed"] is False
    assert parts[1]["content"] == "```js\nconst Ａ = 1;"
    assert parts[1]["code"] == "const Ａ = 1;"


def test_multiple_code_regions_are_all_detected() -> None:
    text = "甲`一`乙\n```py\n二\n```\n丙`三`丁"

    parts = split_code_regions(text)
    types = [part["type"] for part in parts]

    assert types == [
        "text",
        "inline_code",
        "text",
        "fenced_code",
        "text",
        "inline_code",
        "text",
    ]
    assert [part["content"] for part in parts if part["type"] == "inline_code"] == [
        "`一`",
        "`三`",
    ]

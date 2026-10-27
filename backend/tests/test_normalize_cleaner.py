from app.ingestion.cleaning import CleanerPipeline
from app.ingestion.cleaning.normalize import NormalizeCleaner


def test_removes_bom_and_normalizes_all_newline_styles() -> None:
    raw = "\ufeffA\r\nB\rC\nD\ufeff"

    assert NormalizeCleaner().clean(raw) == "A\nB\nC\nD"


def test_converts_fullwidth_ascii_and_ideographic_space() -> None:
    raw = "ＡＢＣ１２３！\u3000"

    assert NormalizeCleaner().clean(raw) == "ABC123! "


def test_removes_zero_width_space() -> None:
    assert NormalizeCleaner().clean("A\u200bB") == "AB"


def test_does_not_apply_broad_nfkc_normalization() -> None:
    assert NormalizeCleaner().clean("① ㎡") == "① ㎡"


def test_preserves_character_content_inside_code_regions() -> None:
    raw = "正文Ａ `行内Ａ`\n```js\nconst Ａ = 'Ａ';\n```"

    expected = "正文A `行内Ａ`\n```js\nconst Ａ = 'Ａ';\n```"
    assert NormalizeCleaner().clean(raw) == expected


def test_normalizes_newlines_inside_code_while_preserving_code_characters() -> None:
    raw = "A\r\n```js\r\nconst Ａ = 1;\r\n```"

    expected = "A\n```js\nconst Ａ = 1;\n```"
    assert NormalizeCleaner().clean(raw) == expected


def test_preserve_code_false_allows_character_cleaning_inside_code() -> None:
    raw = "正文Ａ `行内Ａ`\n```js\nconst Ａ = 1;\n```"

    expected = "正文A `行内A`\n```js\nconst A = 1;\n```"
    assert NormalizeCleaner(preserve_code=False).clean(raw) == expected


def test_individual_rules_can_be_disabled() -> None:
    raw = "\ufeffＡ\u200b\r\nB"

    assert NormalizeCleaner(remove_bom=False).clean(raw) == "\ufeffA\nB"
    assert NormalizeCleaner(normalize_newlines=False).clean(raw) == "A\r\nB"
    assert NormalizeCleaner(remove_zero_width=False).clean(raw) == "A\u200b\nB"
    assert NormalizeCleaner(normalize_fullwidth=False).clean(raw) == "Ａ\nB"


def test_empty_string_returns_empty_string() -> None:
    assert NormalizeCleaner().clean("") == ""


def test_cleaning_is_idempotent() -> None:
    cleaner = NormalizeCleaner()
    raw = "\ufeffＡＰＩ\r\n正文\u200b\r\n```js\r\nconst Ａ = 1;\r\n```"

    cleaned_once = cleaner.clean(raw)
    cleaned_twice = cleaner.clean(cleaned_once)

    assert cleaned_twice == cleaned_once


def test_normalize_cleaner_works_inside_pipeline() -> None:
    pipeline = CleanerPipeline([NormalizeCleaner()])

    result = pipeline.clean("ＡＰＩ\r\nB")

    assert result.text == "API\nB"
    assert len(result.steps) == 1
    assert result.steps[0].cleaner_name == "normalize"
    assert result.steps[0].before_chars == len("ＡＰＩ\r\nB")
    assert result.steps[0].after_chars == len("API\nB")
import json
from pathlib import Path
import subprocess
import sys

import pytest

from app.ingestion.cleaning import BaseCleaner, CleanerPipeline, build_cleaning_report
from scripts.preview_cleaning import main


class SameLengthCleaner(BaseCleaner):
    name = "same-length"

    def clean(self, text: str) -> str:
        return text.replace("A", "B")


class DropPageLineCleaner(BaseCleaner):
    name = "drop-page-line"

    def clean(self, text: str) -> str:
        return text.replace("第 1 页\n", "")


class LongCleaner(BaseCleaner):
    name = "long"

    def clean(self, text: str) -> str:
        return "B" * 100


def test_trace_is_empty_by_default() -> None:
    result = CleanerPipeline([SameLengthCleaner()]).clean("A")

    assert result.text == "B"
    assert result.trace == ()


def test_capture_trace_records_before_and_after_text() -> None:
    result = CleanerPipeline([SameLengthCleaner()]).clean(
        "A",
        capture_trace=True,
    )

    trace = result.trace[0]
    assert trace.cleaner_name == "same-length"
    assert trace.before_text == "A"
    assert trace.after_text == "B"


def test_same_length_change_is_reported_as_changed() -> None:
    result = CleanerPipeline([SameLengthCleaner()]).clean(
        "A",
        capture_trace=True,
    )

    report = build_cleaning_report("sample.md", "A", result)
    step = report["steps"][0]

    assert step["delta_chars"] == 0
    assert step["changed"] is True
    assert "-A" in step["diff"]
    assert "+B" in step["diff"]


def test_diff_report_requires_trace() -> None:
    result = CleanerPipeline([SameLengthCleaner()]).clean("A")

    with pytest.raises(ValueError, match="capture_trace=True"):
        build_cleaning_report("sample.md", "A", result)


def test_summary_only_report_does_not_require_trace() -> None:
    result = CleanerPipeline([SameLengthCleaner()]).clean("A")

    report = build_cleaning_report(
        "sample.md",
        "A",
        result,
        include_diff=False,
    )

    assert report["steps"][0]["changed"] is True
    assert report["steps"][0]["diff"] is None


def test_diff_shows_removed_page_line() -> None:
    raw = "正文\n第 1 页\n下一段"
    result = CleanerPipeline([DropPageLineCleaner()]).clean(
        raw,
        capture_trace=True,
    )

    report = build_cleaning_report("sample.md", raw, result)

    assert "第 1 页" in report["steps"][0]["diff"]


def test_diff_is_truncated() -> None:
    result = CleanerPipeline([LongCleaner()]).clean(
        "A",
        capture_trace=True,
    )

    report = build_cleaning_report(
        "sample.md",
        "A",
        result,
        max_diff_chars=20,
    )

    assert report["steps"][0]["diff"].endswith("... [diff truncated]")
    assert len(report["steps"][0]["diff"]) <= 20 + len("\n... [diff truncated]")


def test_report_is_json_serializable() -> None:
    result = CleanerPipeline([SameLengthCleaner()]).clean(
        "A",
        capture_trace=True,
    )

    report = build_cleaning_report("sample.md", "A", result)

    payload = json.dumps(report, ensure_ascii=False)
    assert '"cleaner": "same-length"' in payload


def test_invalid_max_diff_chars_is_rejected() -> None:
    result = CleanerPipeline([]).clean("")

    with pytest.raises(ValueError, match="大于 0"):
        build_cleaning_report("sample.md", "", result, max_diff_chars=0)

    with pytest.raises(TypeError, match="整数"):
        build_cleaning_report("sample.md", "", result, max_diff_chars=1.5)


def test_preview_script_writes_json_report(tmp_path: Path) -> None:
    path = tmp_path / "preview.md"
    path.write_text("正文\n第 1 页\n下一段", encoding="utf-8")
    output = tmp_path / "reports" / "preview.json"

    exit_code = main([str(path), "--output", str(output)])

    assert exit_code == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["source"] == str(path)
    assert [step["cleaner"] for step in report["steps"]] == [
        "normalize",
        "media",
        "noise",
        "whitespace",
        "dedup",
    ]
    noise_step = next(step for step in report["steps"] if step["cleaner"] == "noise")
    assert "第 1 页" in noise_step["diff"]


def test_preview_script_supports_summary_only(tmp_path: Path) -> None:
    path = tmp_path / "preview.md"
    path.write_text("正文\n第 1 页\n下一段", encoding="utf-8")
    output = tmp_path / "preview.json"

    exit_code = main([str(path), "--output", str(output), "--no-diff"])

    assert exit_code == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert all(step["diff"] is None for step in report["steps"])


def test_preview_script_returns_error_for_missing_file(tmp_path: Path) -> None:
    assert main([str(tmp_path / "missing.md")]) == 2
def test_preview_script_runs_as_a_file(tmp_path: Path) -> None:
    path = tmp_path / "preview.md"
    path.write_text("正文\n第 1 页\n下一段", encoding="utf-8")
    backend_root = Path(__file__).resolve().parents[1]

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/preview_cleaning.py",
            str(path),
            "--no-diff",
        ],
        cwd=backend_root,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    report = json.loads(completed.stdout)
    assert report["source"] == str(path)
    assert report["steps"][-1]["cleaner"] == "dedup"
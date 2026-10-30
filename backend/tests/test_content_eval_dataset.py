import json
from pathlib import Path


EVALUATION_DIR = Path(__file__).resolve().parents[1] / "evaluation"


def load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_content_retrieval_dataset_structure() -> None:
    cases = load_jsonl(EVALUATION_DIR / "retrieval_cases_content.jsonl")

    assert len(cases) == 30
    assert len({case["id"] for case in cases}) == len(cases)
    assert {case["category"] for case in cases} >= {
        "semantic",
        "exact",
        "numeric",
        "multi_snippet",
    }
    for case in cases:
        assert case["question"].strip()
        assert case["relevant_doc"]
        assert not case["relevant_doc"].startswith(("/", "\\"))
        assert "chunk_index" not in case
        assert case["relevant_snippets"]
        assert all(snippet.strip() for snippet in case["relevant_snippets"])


def test_refusal_dataset_structure() -> None:
    cases = load_jsonl(EVALUATION_DIR / "qa_refusal_cases.jsonl")

    assert len(cases) == 3
    assert len({case["id"] for case in cases}) == len(cases)
    for case in cases:
        assert case["question"].strip()
        assert case["expected_behavior"] == "refuse"
        assert case["reason"].strip()
"""校验内容锚点评测集中的 snippet 是否存在于对应语料文档。"""

import argparse
import json
import sys
import unicodedata
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.ingestion.cleaning import PdfPageNoiseCleaner
from app.ingestion.cleaning.factory import build_default_cleaner_pipeline
from app.ingestion.loader import parse_document


def normalize_match_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text)
    return "".join(normalized.split())


def load_cases(path: Path) -> list[dict]:
    cases = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8-sig").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        try:
            case = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"第 {line_number} 行 JSON 错误: {exc}") from exc
        cases.append(case)
    return cases


def clean_corpus_document(path: Path) -> str:
    parsed = parse_document(path)
    text = parsed.text
    if parsed.metadata["file_type"] == "pdf":
        text = PdfPageNoiseCleaner().clean_pages(parsed.pages).text
    return build_default_cleaner_pipeline().clean(text).text


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="校验内容锚点评测集")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--corpus-root", type=Path, required=True)
    args = parser.parse_args(argv)

    cases = load_cases(args.dataset)
    seen_ids: set[str] = set()
    cache: dict[str, str] = {}
    failures: list[str] = []

    for case in cases:
        case_id = case.get("id", "<missing-id>")
        if case_id in seen_ids:
            failures.append(f"{case_id}: 重复 id")
        seen_ids.add(case_id)

        doc_name = case.get("relevant_doc")
        snippets = case.get("relevant_snippets")
        if not isinstance(doc_name, str) or not doc_name:
            failures.append(f"{case_id}: relevant_doc 缺失")
            continue
        if not isinstance(snippets, list) or not snippets:
            failures.append(f"{case_id}: relevant_snippets 缺失")
            continue

        path = args.corpus_root / doc_name
        if not path.exists():
            failures.append(f"{case_id}: 文件不存在 {path}")
            continue

        if doc_name not in cache:
            try:
                cache[doc_name] = normalize_match_text(clean_corpus_document(path))
            except Exception as exc:
                failures.append(f"{case_id}: 文档清洗失败 {doc_name}: {exc}")
                continue

        matched_text = cache[doc_name]
        for index, snippet in enumerate(snippets):
            if not isinstance(snippet, str) or not snippet.strip():
                failures.append(f"{case_id}: snippet[{index}] 为空")
                continue
            if normalize_match_text(snippet) not in matched_text:
                failures.append(
                    f"{case_id}: snippet[{index}] 不存在于 {doc_name}: {snippet}"
                )

    print(f"评测集: {args.dataset}")
    print(f"样本数: {len(cases)}")
    print(f"涉及文档数: {len(cache)}")
    print(f"失败数: {len(failures)}")
    for failure in failures:
        print(f"- {failure}")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
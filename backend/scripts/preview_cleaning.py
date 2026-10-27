import argparse
import json
from pathlib import Path
import sys

# 直接执行 scripts/preview_cleaning.py 时，确保 backend 根目录位于导入路径中。
BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.ingestion.cleaning.factory import build_default_cleaner_pipeline
from app.ingestion.cleaning.report import build_cleaning_report
from app.ingestion.loader import load_document


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="预览文档清洗每一步的变化")
    parser.add_argument("path", type=Path, help="待清洗的 .txt / .md / .pdf 文件")
    parser.add_argument("--output", type=Path, default=None, help="可选 JSON 报告输出路径")
    parser.add_argument("--no-diff", action="store_true", help="只输出摘要，不生成 diff")
    parser.add_argument(
        "--max-diff-chars",
        type=int,
        default=12000,
        help="每一步 diff 的最大字符数",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

    args = parse_args(argv)
    if not args.path.exists():
        print(f"文件不存在: {args.path}", file=sys.stderr)
        return 2

    try:
        content, metadata = load_document(path=args.path)
        pipeline = build_default_cleaner_pipeline()
        clean_result = pipeline.clean(
            text=content,
            capture_trace=not args.no_diff,
        )
        report = build_cleaning_report(
            source=str(args.path),
            raw_text=content,
            result=clean_result,
            include_diff=not args.no_diff,
            max_diff_chars=args.max_diff_chars,
        )
    except (OSError, ValueError, TypeError) as exc:
        print(f"清洗预览失败: {exc}", file=sys.stderr)
        return 2

    report_json = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output is None:
        print(report_json)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report_json, encoding="utf-8")
        print(f"清洗报告已写入: {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
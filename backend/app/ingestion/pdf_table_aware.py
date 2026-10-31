"""表格感知的 PDF 解析与分块实验实现。"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pdfplumber

from app.ingestion.cleaning import PdfPageNoiseCleaner
from app.ingestion.cleaning.factory import build_default_cleaner_pipeline
from app.ingestion.loader import ParsedPage
from app.ingestion.pdf_text import dewrap_pdf_pages
from app.ingestion.splitter import ChunkStrategy, split_text


@dataclass(frozen=True)
class PdfBlock:
    block_type: str
    text: str
    page_number: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TableAwareChunk:
    text: str
    page_number: int
    block_type: str
    metadata: dict[str, Any] = field(default_factory=dict)


def _normalize_cell(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).replace("\r\n", "\n").replace("\r", "\n")
    text = " ".join(text.split())
    return text.replace("|", "\\|")


def table_to_markdown(
    rows: list[list[Any]],
    *,
    page_number: int,
    table_index: int,
) -> str:
    """把 pdfplumber 二维表格转换为 Markdown。"""
    normalized = [
        [_normalize_cell(cell) for cell in row]
        for row in rows
        if row and any(cell not in (None, "") for cell in row)
    ]
    if not normalized:
        return ""

    width = max(len(row) for row in normalized)
    normalized = [row + [""] * (width - len(row)) for row in normalized]
    header = normalized[0]
    body = normalized[1:]
    separator = ["---"] * width

    lines = [
        f"[表格 第{page_number}页 表{table_index}]",
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(separator) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in body)
    return "\n".join(lines)


def _outside_table_filter(boxes: list[tuple[float, float, float, float]]):
    def predicate(obj: dict[str, Any]) -> bool:
        if obj.get("object_type") != "char":
            return True
        x = (obj["x0"] + obj["x1"]) / 2
        y = (obj["top"] + obj["bottom"]) / 2
        return all(
            not (x0 <= x <= x1 and top <= y <= bottom)
            for x0, top, x1, bottom in boxes
        )

    return predicate


def parse_table_aware_pdf(path: Path) -> tuple[PdfBlock, ...]:
    """提取 PDF 的正文块和表格块，并清理正文页眉页脚。"""
    prose_pages: list[ParsedPage] = []
    table_blocks: list[PdfBlock] = []

    with pdfplumber.open(path) as pdf:
        for page_index, page in enumerate(pdf.pages, start=1):
            tables = page.find_tables()
            boxes = [table.bbox for table in tables]
            filtered_page = page.filter(_outside_table_filter(boxes)) if boxes else page
            prose_text = filtered_page.extract_text() or ""
            prose_pages.append(ParsedPage(page_number=page_index, text=prose_text))

            for table_index, table in enumerate(tables, start=1):
                rows = table.extract() or []
                markdown = table_to_markdown(
                    rows,
                    page_number=page_index,
                    table_index=table_index,
                )
                if markdown:
                    table_blocks.append(
                        PdfBlock(
                            block_type="table",
                            text=markdown,
                            page_number=page_index,
                            metadata={
                                "table_index": table_index,
                                "table_id": f"p{page_index}-t{table_index}",
                                "row_count": max(len(rows) - 1, 0),
                            },
                        )
                    )

    page_clean_result = PdfPageNoiseCleaner().clean_pages(prose_pages)
    dewrapped_pages = dewrap_pdf_pages(page_clean_result.pages)
    cleaner = build_default_cleaner_pipeline()
    blocks: list[PdfBlock] = []
    for page in dewrapped_pages:
        cleaned = cleaner.clean(page.text).text.strip()
        if cleaned:
            blocks.append(
                PdfBlock(
                    block_type="text",
                    text=cleaned,
                    page_number=page.page_number,
                )
            )
    blocks.extend(table_blocks)
    return tuple(blocks)


def split_markdown_table(
    markdown: str,
    *,
    chunk_size: int,
    metadata: dict[str, Any],
) -> list[TableAwareChunk]:
    """超长表格按数据行切分，并在每个 chunk 重复表头和标题。"""
    lines = markdown.splitlines()
    if len(lines) < 3:
        return []

    prefix = "\n".join(lines[:3])
    data_lines = lines[3:]
    if not data_lines or len(markdown) <= chunk_size:
        return [
            TableAwareChunk(
                text=markdown,
                page_number=int(metadata.get("page_number", 0)),
                block_type="table",
                metadata={**metadata, "row_start": 1, "row_end": len(data_lines)},
            )
        ]

    chunks: list[TableAwareChunk] = []
    current_rows: list[str] = []
    start_row = 1
    for row_index, row in enumerate(data_lines, start=1):
        candidate_rows = current_rows + [row]
        candidate = prefix + "\n" + "\n".join(candidate_rows)
        if current_rows and len(candidate) > chunk_size:
            chunks.append(
                TableAwareChunk(
                    text=prefix + "\n" + "\n".join(current_rows),
                    page_number=int(metadata.get("page_number", 0)),
                    block_type="table",
                    metadata={**metadata, "row_start": start_row, "row_end": row_index - 1},
                )
            )
            current_rows = [row]
            start_row = row_index
        else:
            current_rows = candidate_rows

    if current_rows:
        chunks.append(
            TableAwareChunk(
                text=prefix + "\n" + "\n".join(current_rows),
                page_number=int(metadata.get("page_number", 0)),
                block_type="table",
                metadata={**metadata, "row_start": start_row, "row_end": len(data_lines)},
            )
        )
    return chunks


def split_table_aware_blocks(
    blocks: tuple[PdfBlock, ...],
    *,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    strategy: ChunkStrategy = "fixed",
) -> list[TableAwareChunk]:
    """正文按指定策略切分，表格按行切分并重复表头。"""
    chunks: list[TableAwareChunk] = []
    for block in blocks:
        if block.block_type == "table":
            chunks.extend(
                split_markdown_table(
                    block.text,
                    chunk_size=chunk_size,
                    metadata={
                        **block.metadata,
                        "page_number": block.page_number,
                    },
                )
            )
            continue

        for text in split_text(
            block.text,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            strategy=strategy,
        ):
            chunks.append(
                TableAwareChunk(
                    text=text,
                    page_number=block.page_number,
                    block_type="text",
                    metadata={},
                )
            )
    return chunks
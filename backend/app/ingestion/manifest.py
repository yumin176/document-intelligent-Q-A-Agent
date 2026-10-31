"""ingestion manifest 读写工具。"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def save_manifest(
    path: Path,
    *,
    collection_name: str,
    embedding_model: str,
    embedding_base_url: str,
    embedding_dimension: int,
    chunk_strategy: str,
    chunk_size: int,
    chunk_overlap: int,
    documents: list[dict[str, Any]],
) -> dict[str, Any]:
    """把一次批量入库的配置和文档清单写入 JSON。"""
    manifest = {
        "collection_name": collection_name,
        "embedding_model": embedding_model,
        "embedding_base_url": embedding_base_url,
        "embedding_dimension": embedding_dimension,
        "chunk_strategy": chunk_strategy,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "documents": documents,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest


def load_manifest(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))
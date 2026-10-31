import json
from pathlib import Path

from app.ingestion.manifest import load_manifest, save_manifest
from scripts.ingest_corpus import main


def test_manifest_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    documents = [
        {
            "doc_name": "docs/a.md",
            "document_id": "doc-a",
            "chunk_count": 3,
            "content_hash": "abc",
            "ingested_at": "2026-10-30T00:00:00+00:00",
        }
    ]

    manifest = save_manifest(
        path,
        collection_name="qa_chunks_prod_v1",
        embedding_model="qwen3.7-text-embedding",
        embedding_base_url="https://example.com/v1",
        embedding_dimension=1024,
        chunk_strategy="fixed",
        chunk_size=500,
        chunk_overlap=50,
        documents=documents,
    )

    loaded = load_manifest(path)

    assert loaded == manifest
    assert loaded["collection_name"] == "qa_chunks_prod_v1"
    assert loaded["embedding_dimension"] == 1024
    assert loaded["documents"] == documents


def test_ingest_corpus_dry_run_writes_manifest(tmp_path: Path) -> None:
    corpus_root = tmp_path / "corpus"
    corpus_root.mkdir()
    (corpus_root / "a.md").write_text("正文内容", encoding="utf-8")
    output = tmp_path / "manifest.json"

    exit_code = main(
        [
            "--corpus-root",
            str(corpus_root),
            "--docs",
            "a.md",
            "--strategy",
            "fixed",
            "--dry-run",
            "--manifest",
            str(output),
        ]
    )

    assert exit_code == 0
    manifest = json.loads(output.read_text(encoding="utf-8"))
    assert manifest["collection_name"] == "dry-run"
    assert manifest["chunk_strategy"] == "fixed"
    assert len(manifest["documents"]) == 1
    assert manifest["documents"][0]["doc_name"] == "a.md"
    assert manifest["documents"][0]["chunk_count"] >= 1
    assert manifest["embedding_dimension"] == 8
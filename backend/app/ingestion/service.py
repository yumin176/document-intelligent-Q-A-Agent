import unicodedata
from hashlib import sha1, sha256
from pathlib import Path

from pydantic import BaseModel

from app.config import get_settings
from app.embedding.base import Embedder
from app.ingestion.cleaning import CleanerPipeline
from app.ingestion.cleaning.factory import build_default_cleaner_pipeline
from app.ingestion.cleaning.pdf_page_noise import PdfPageNoiseCleaner
from app.ingestion.context import build_chunk_header, nearest_heading
from app.ingestion.loader import parse_document
from app.ingestion.pdf_text import dewrap_pdf_pages, join_pdf_pages
from app.ingestion.splitter import ChunkStrategy, split_text
from app.vectorstore.base import ChunkRecord, VectorStore


class IngestResult(BaseModel):
    document_id: str
    filename: str
    chunk_count: int


def normalize_doc_name(doc_name: str) -> str:
    """统一语料库相对路径，避免同名文件和不同路径分隔符造成 ID 冲突。"""
    normalized = unicodedata.normalize("NFKC", doc_name.strip()).replace("\\", "/")
    while normalized.startswith("./"):
        normalized = normalized[2:]
    return normalized.casefold()


def normalize_filename(filename: str) -> str:
    """兼容旧调用；filename 也按文档名规则规范化。"""
    return normalize_doc_name(filename)

# 创建chunk锚点
def build_anchor_id(
    document_id: str,
    chunk_index: int,
    chunk_text: str,
) -> str:
    prefix_hash = sha1(
        chunk_text[:50].encode("utf-8")
    ).hexdigest()[:8]
    # 同一文档 + 相同分块配置 + 内容未变化
    return f"{document_id}#{chunk_index}:{prefix_hash}"

def build_chunk_records(
    text: str,
    file_meta: dict,
    embedder: Embedder,
    source: str,
    chunk_strategy: ChunkStrategy = "fixed",
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    header_context: bool = False,
) -> list[ChunkRecord]:
    """把文档分块、向量化并组装成可入库记录。

    header_context 为 True 时，每个块前会加上 `【公司｜年份｜页码】`
    （正文块用最近章节标题），再送去 embedding。见 app/ingestion/context.py。
    """
    chunks = split_text(
        text=text,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        strategy=chunk_strategy,
    )

    filename = file_meta["filename"]
    doc_name = file_meta.get("doc_name") or filename
    identity_key = normalize_doc_name(doc_name)
    document_id = sha256(identity_key.encode("utf-8")).hexdigest()[:32]
    content_hash = sha256(text.encode("utf-8")).hexdigest()

    # 待向量化的文本；开启上下文时与入库文本一致，否则就是原始块
    embed_texts: list[str] = []
    if header_context:
        cursor = 0
        for chunk in chunks:
            position = text.find(chunk, cursor)
            if position < 0:
                position = cursor
            cursor = position + 1
            embed_texts.append(
                build_chunk_header(
                    doc_name,
                    heading=nearest_heading(text, position),
                )
                + chunk
            )
    else:
        embed_texts = list(chunks)

    embeddings = embedder.embed_batch(embed_texts)

    if len(chunks) != len(embeddings):
        raise RuntimeError(
            f"分块数量 {len(chunks)} 与向量数量 {len(embeddings)} 不一致"
        )

    # 兼容直接调用该函数的测试或脚本；没有清洗摘要时回退到当前文本长度。
    raw_num_chars = file_meta.get("raw_num_chars", len(text))
    num_chars = file_meta.get("num_chars", len(text))
    cleaning_steps = list(file_meta.get("cleaning_steps", []))
    page_cleaning = dict(file_meta.get("page_cleaning", {}))
    embedding_dimension = len(embeddings[0]) if embeddings else 0

    records: list[ChunkRecord] = []
    for index, (chunk, embed_text, embedding) in enumerate(
        zip(chunks, embed_texts, embeddings)
    ):
        # 锚点 id 与哈希仍基于原始块，保证开关上下文前后 id 稳定可比。
        anchor_id = build_anchor_id(document_id, index, chunk)
        chunk_hash = sha256(chunk.encode("utf-8")).hexdigest()
        records.append(
            ChunkRecord(
                # 主键改成稳定、可定位的锚点 id。
                id=anchor_id,
                text=embed_text,
                embedding=embedding,
                metadata={
                    "anchor_id": anchor_id,
                    "doc_name": doc_name,
                    "chunk_hash": chunk_hash,
                    "document_id": document_id,
                    "content_hash": content_hash,
                    "source": source,
                    "filename": filename,
                    "file_type": file_meta["file_type"],
                    "chunk_index": index,
                    "raw_num_chars": raw_num_chars,
                    "num_chars": num_chars,
                    "cleaning_steps": cleaning_steps,
                    "page_cleaning": page_cleaning,
                    "chunk_strategy": chunk_strategy,
                    "chunk_size": chunk_size,
                    "chunk_overlap": chunk_overlap,
                    "header_context": header_context,
                    "embedding_model": file_meta.get("embedding_model", ""),
                    "embedding_base_url": file_meta.get("embedding_base_url", ""),
                    "embedding_dimension": embedding_dimension,
                },
            )
        )
    return records


class IngestionService:
    """离线索引：文档解析 → 页级清洗 → 文本清洗 → 分块 → 向量化 → 入库。"""

    def __init__(
        self,
        embedder: Embedder,
        vector_store: VectorStore,
        cleaner: CleanerPipeline | None = None,
        page_cleaner: PdfPageNoiseCleaner | None = None,
        chunk_strategy: ChunkStrategy | None = None,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        header_context: bool = False,
    ) -> None:
        settings = get_settings()
        self.embedder = embedder
        self.vector_store = vector_store
        self.cleaner = cleaner if cleaner is not None else build_default_cleaner_pipeline()
        self.page_cleaner = (
            page_cleaner if page_cleaner is not None else PdfPageNoiseCleaner()
        )
        self.chunk_strategy = (
            settings.chunk_strategy if chunk_strategy is None else chunk_strategy
        )
        self.chunk_size = settings.chunk_size if chunk_size is None else chunk_size
        self.chunk_overlap = (
            settings.chunk_overlap if chunk_overlap is None else chunk_overlap
        )
        self.header_context = header_context

    def ingest_file(
        self,
        path: Path,
        original_filename: str | None = None,
        doc_name: str | None = None,
    ) -> IngestResult:
        parsed = parse_document(path)
        # 原始文本在页级清洗前保存，用于记录真实的 raw_num_chars。
        raw_text = parsed.text
        file_meta = parsed.metadata

        # 旧入口已改为 parse_document，以便保留 PDF 页边界。
        # raw_text, file_meta = load_document(path)

        page_cleaning = {"enabled": False}
        text_before_generic_cleaning = raw_text

        if file_meta["file_type"] == "pdf":
            page_clean_result = self.page_cleaner.clean_pages(parsed.pages)
            dewrapped_pages = dewrap_pdf_pages(page_clean_result.pages)
            text_before_generic_cleaning = join_pdf_pages(dewrapped_pages)
            page_cleaning = {
                "enabled": True,
                "before_chars": len(raw_text),
                "after_chars": len(text_before_generic_cleaning),
                "removed_page_numbers": page_clean_result.removed_page_numbers,
                "removed_header_keys": list(page_clean_result.removed_header_keys),
                "removed_footer_keys": list(page_clean_result.removed_footer_keys),
            }

        clean_result = self.cleaner.clean(text_before_generic_cleaning)
        cleaned_text = clean_result.text
        if not cleaned_text.strip():
            raise ValueError("清洗后文档为空，无法入库")

        if original_filename:
            safe_filename = Path(original_filename).name
            file_meta["filename"] = safe_filename
            source = safe_filename
            if doc_name is None:
                doc_name = safe_filename
        elif doc_name is None:
            doc_name = file_meta["filename"]
            source = str(path.resolve())
        else:
            source = doc_name

        file_meta["doc_name"] = doc_name
        file_meta["embedding_model"] = getattr(
            self.embedder, "model", get_settings().embedding_model
        )
        file_meta["embedding_base_url"] = getattr(
            self.embedder, "base_url", get_settings().embedding_base_url
        )

        # 增加源数据
        file_meta["raw_num_chars"] = len(raw_text)
        file_meta["num_chars"] = len(cleaned_text)
        file_meta["page_cleaning"] = page_cleaning
        file_meta["cleaning_steps"] = (
            (["pdf_page_noise"] if page_cleaning["enabled"] else [])
            + [step.cleaner_name for step in clean_result.steps]
        )

        records = build_chunk_records(
            text=cleaned_text,
            file_meta=file_meta,
            embedder=self.embedder,
            source=source,
            chunk_strategy=self.chunk_strategy,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            header_context=self.header_context,
        )
        self.vector_store.upsert(records)

        document_id = records[0].metadata["document_id"] if records else ""
        if document_id:
            # 安全替换：先写入新 chunk，再删除同一文档中已不存在的旧锚点。
            new_ids = {record.id for record in records}
            old_ids = {
                record.id
                for record in self.vector_store.all_chunks()
                if record.metadata.get("document_id") == document_id
            }
            stale_ids = old_ids - new_ids
            if stale_ids:
                self.vector_store.delete_by_ids(list(stale_ids))
        return IngestResult(
            document_id=document_id,
            filename=file_meta["filename"],
            chunk_count=len(records),
        )
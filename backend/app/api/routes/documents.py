import shutil
import tempfile
from collections.abc import Callable
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from app.dependencies import get_bm25_refresher, get_ingestion_service
from app.ingestion.service import IngestionService
from app.schemas import DocumentUploadResponse

router = APIRouter(prefix="/api", tags=["documents"])


@router.post("/documents", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    service: IngestionService = Depends(get_ingestion_service),
    refresh_bm25: Callable[[], None] = Depends(get_bm25_refresher),
) -> DocumentUploadResponse:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in {".txt", ".md", ".pdf"}:
        raise HTTPException(415, "仅支持 .txt / .md / .pdf 文件")

    temp_path = Path(tempfile.NamedTemporaryFile(delete=False, suffix=suffix).name)

    try:
        with temp_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        ingest_result = service.ingest_file(
            path=temp_path,
            original_filename=file.filename,
        )
        refresh_bm25()

        return DocumentUploadResponse(
            document_id=ingest_result.document_id,
            filename=ingest_result.filename,
            chunk_count=ingest_result.chunk_count,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        temp_path.unlink(missing_ok=True)

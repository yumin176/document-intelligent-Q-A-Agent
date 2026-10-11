import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from app.dependencies import get_ingestion_service
from app.ingestion.service import IngestionService
from app.schemas import DocumentUploadResponse
import shutil
router = APIRouter(prefix="/api", tags=["documents"])

# 上传文件接口
@router.post("/documents", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    service: IngestionService = Depends(get_ingestion_service),
) -> DocumentUploadResponse:
    # 1、判断文件类型是不是txt、md、pdf
    # 踩坑，file.content_type获取的不是文件拓展名，要用suffix来获取
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in {".txt", ".md", ".pdf"}:
        raise HTTPException(415, "仅支持 .txt / .md / .pdf 文件")
    # 2、创建临时文件
    temp_path = Path(
        tempfile.NamedTemporaryFile(delete=False, suffix=suffix).name
    )
    try:
        # 3、写入临时文件
        with temp_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        # 4、离线索引
        ingest_result = service.ingest_file(path=temp_path)
        return DocumentUploadResponse(
            document_id=ingest_result.document_id,
            filename=ingest_result.filename,
            chunk_count=ingest_result.chunk_count,
        )
    finally:
        # 5、删除临时文件
        temp_path.unlink(missing_ok=True)

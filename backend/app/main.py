"""FastAPI 应用入口。"""

import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes.chat import router as chat_router
from app.api.routes.documents import router as documents_router
from app.api.routes.health import router as health_router
from app.api.routes.knowledge import router as knowledge_router
from app.config import get_settings
from app.observability.tracing import is_enabled

FRONTEND_DIR = Path(__file__).resolve().parents[2] / "frontend"

logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    """创建应用实例。

    使用工厂函数便于测试时创建独立实例；
    静态资源与页面路由也在这里注册，保证测试实例与生产实例行为一致。
    """
    settings = get_settings()
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
    )
    app.include_router(health_router)
    app.include_router(chat_router)
    app.include_router(documents_router)
    app.include_router(knowledge_router)

    logger.info("LangSmith 追踪: %s", "已启用" if is_enabled() else "关闭")

    app.mount(
        "/static",
        StaticFiles(directory=FRONTEND_DIR),
        name="static",
    )

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(FRONTEND_DIR / "index.html")

    @app.get("/knowledge")
    async def knowledge() -> FileResponse:
        """知识库只读查看页面：文档列表 → chunk 列表 → 锚点定位。"""
        return FileResponse(FRONTEND_DIR / "knowledge.html")

    return app


app = create_app()

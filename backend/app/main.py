"""FastAPI 应用入口。"""

from fastapi import FastAPI

from app.api.routes.health import router as health_router
from app.api.routes.chat import router as chat_router
from app.config import get_settings
from app.api.routes.documents import router as documents_router
from pathlib import Path

from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

def create_app() -> FastAPI:
    """创建应用实例。

    使用工厂函数便于测试时创建独立实例，
    后续接入检索、Agent 等路由时都从这里统一注册。
    """
    settings = get_settings()
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
    )
    app.include_router(health_router)
    app.include_router(chat_router)
    app.include_router(documents_router)
    return app

app = create_app()


FRONTEND_DIR = Path(__file__).resolve().parents[2] / "frontend"

app.mount(
    "/static",
    StaticFiles(directory=FRONTEND_DIR),
    name="static",
)

@app.get("/")
async def index() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "index.html")

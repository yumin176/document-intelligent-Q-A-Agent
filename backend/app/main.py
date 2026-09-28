"""FastAPI 应用入口。"""

from fastapi import FastAPI

from app.api.routes.health import router as health_router
from app.config import get_settings


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
    return app


app = create_app()
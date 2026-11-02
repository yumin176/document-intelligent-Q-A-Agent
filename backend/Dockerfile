# 智答 Agent 后端镜像
# 构建上下文为仓库根目录：docker build -f backend/Dockerfile .
# 目录布局说明：main.py 通过 parents[2] 定位 frontend 目录，
# 因此镜像内保持 <root>/backend/app + <root>/frontend 的相对结构。
# 基础镜像走华为云镜像源（与 compose 里的 Milvus 同源）：
# Docker Desktop 默认的 registry 镜像加速器在本机证书校验失败，直连 docker.io 又太慢。
FROM swr.cn-north-4.myhuaweicloud.com/ddn-k8s/docker.io/library/python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# 依赖单独一层：改代码不触发重装
COPY backend/requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt && rm /tmp/requirements.txt

COPY backend/app /srv/backend/app
COPY frontend /srv/frontend

# 非 root 用户运行
RUN useradd --create-home appuser && chown -R appuser:appuser /srv
USER appuser

WORKDIR /srv/backend

EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=5s --start-period=20s --retries=5 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

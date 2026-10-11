# 智答 Agent（document-intelligent-Q-A-Agent）

基于 **混合检索 + Agent 自主推理** 的文档智能问答系统。

> 目标不是复刻一个 PDF 问答 Demo，而是搭建一个具备生产级工程能力的 RAG 应用：
> 文档入库 → 向量检索 → LLM 生成带引用的回答，并对检索与生成过程做可观测、可评测。

## 当前进度

| 里程碑 | 内容 | 状态 |
|---|---|---|
| M0 | 工程骨架、配置、测试、VSCode 配置 | ✅ 已完成 |
| M1 | 文档解析、分块、Embedding、Milvus 入库与检索 | ✅ 已完成 |
| M2 | LLM 接入层、RAG 组装服务、引用溯源 | ✅ 已完成 |
| M3 | FastAPI 接口、SSE 流式输出、前端对话界面 | ⏳ 进行中 |
| M4 | 混合检索（向量 + BM25 + Rerank）与对比评测 | ⏳ 待开始 |
| M5 | LangGraph Agent 工作流（Router / Grader / Rewrite） | ⏳ 待开始 |
| M6 | LangSmith 追踪、检索评测、Docker 打包收尾 | ⏳ 待开始 |

## 技术栈

- 后端：FastAPI
- LLM 框架：OpenAI 兼容 SDK（可切换 DeepSeek / 百炼 / OpenAI 等）
- 向量库：Milvus（Docker Compose 部署）
- Embedding：OpenAI 兼容接口（默认百炼 `qwen3.7-text-embedding`）
- LLM：OpenAI 兼容接口（默认 `deepseek-v4-flash`）
- 依赖管理：Python venv + `requirements.txt`
- 测试：pytest

## 核心特性

- 文档解析：支持 `.txt` / `.md` / `.pdf`
- 文本分块：可配置 `chunk_size` 与 `chunk_overlap`
- 向量化与入库：批量调用 Embedding API，写入 Milvus
- 语义检索：基于余弦相似度的 top-k 检索
- RAG 生成：检索结果编号注入 Prompt，LLM 依据资料回答
- 引用溯源：答案标注 `[1]`，返回 `filename`、`chunk_index`、`score`
- 接口与实现分离：`Embedder` / `VectorStore` / `ChatModel` 均可替换

## 系统架构

```text
┌──────────────────────────────────────────────────────┐
│                      文档入库                          │
│  文件 → loader → splitter → Embedder → MilvusStore     │
└──────────────────────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────┐
│                       RAG 问答                         │
│  问题 → Embedder → Milvus 检索 → 组装 Prompt → LLM     │
│                              │                         │
│                              ▼                         │
│                     回答 + 引用来源                     │
└──────────────────────────────────────────────────────┘
```

### 依赖组件

```text
Milvus 容器
├── etcd     保存元数据
└── MinIO    保存向量数据与索引文件
```

- `19530`：pymilvus 连接端口
- `9091`：Milvus 健康检查 / 指标
- `9001`：MinIO 管理控制台

## 目录结构

```text
backend/
├── app/
│   ├── main.py              # FastAPI 入口
│   ├── config.py            # 配置定义与读取
│   ├── api/                 # 接口路由
│   ├── ingestion/           # 文档解析、分块、入库编排
│   ├── embedding/           # Embedding 接口与实现
│   ├── vectorstore/         # 向量库接口与 Milvus 实现
│   ├── llm/                 # LLM 接口与实现
│   ├── rag/                 # RAG 组装服务
│   └── prompts/             # Prompt 模板
├── scripts/
│   ├── inspect_milvus.py    # 只读查看 Milvus 数据
│   ├── ingest_js_note.py    # 文档入库演示
│   └── ask.py               # 命令行 RAG 问答
└── tests/                   # 单元测试与集成测试

docker-compose.yml           # Milvus + etcd + MinIO
.env.example                 # 环境变量模板
```

## 快速开始

### 1. 前置条件

- Python 3.11+
- Docker Desktop（需可运行 Linux 容器，Windows 使用 WSL2）

### 2. 安装 Python 依赖

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 3. 配置环境变量

```powershell
cd backend
Copy-Item ..\.env.example .env
```

编辑 `backend/.env`，至少填写：

```text
EMBEDDING_BASE_URL=你的 Embedding 服务地址
EMBEDDING_API_KEY=你的 Embedding Key
EMBEDDING_MODEL=你的 Embedding 模型名

LLM_BASE_URL=你的 LLM 服务地址
LLM_API_KEY=你的 LLM Key
LLM_MODEL=你的 LLM 模型名
```

> `LLM_BASE_URL` / `LLM_API_KEY` 留空时，会复用 `EMBEDDING_BASE_URL` / `EMBEDDING_API_KEY`。
> `.env` 已被 `.gitignore` 忽略，请勿提交。

### 4. 启动 Milvus

在项目根目录执行：

```powershell
docker compose up -d
```

确认容器健康：

```powershell
docker compose ps
```

### 5. 启动后端（可选）

```powershell
cd backend
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload
```

访问：

- 健康检查：http://127.0.0.1:8000/health
- 接口文档：http://127.0.0.1:8000/docs

### 6. 文档入库

编辑 `backend/scripts/ingest_js_note.py` 中的 `FILE_PATH` 为你的文档路径，然后执行：

```powershell
cd backend
.\.venv\Scripts\python.exe scripts\ingest_js_note.py
```

### 7. 命令行提问

```powershell
cd backend
.\.venv\Scripts\python.exe scripts\ask.py "你的问题" --top-k 3
```

输出包含：回答、引用来源、`score`、命中的原文片段。

### 8. 查看 Milvus 中的数据

```powershell
cd backend
.\.venv\Scripts\python.exe scripts\inspect_milvus.py --limit 10
```

## 配置说明

| 环境变量 | 默认值 | 说明 |
|---|---|---|
| `APP_NAME` | `smart-qa-agent` | 应用名称 |
| `EMBEDDING_BASE_URL` | OpenAI 官方地址 | Embedding 服务地址 |
| `EMBEDDING_API_KEY` | 空 | Embedding API Key |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding 模型名 |
| `EMBEDDING_BATCH_SIZE` | `20` | 每次批量请求的文本条数 |
| `EMBEDDING_TIMEOUT` | `30` | Embedding 请求超时（秒） |
| `LLM_BASE_URL` | 空 | LLM 服务地址，留空复用 Embedding |
| `LLM_API_KEY` | 空 | LLM API Key，留空复用 Embedding |
| `LLM_MODEL` | `qwen-plus` | LLM 模型名 |
| `LLM_TEMPERATURE` | `0` | 生成温度，知识问答建议 0 |
| `LLM_TIMEOUT` | `30` | LLM 请求超时（秒） |
| `MILVUS_URI` | `http://127.0.0.1:19530` | Milvus 连接地址 |
| `MILVUS_COLLECTION_NAME` | `qa_chunks` | 默认 Collection 名称 |
| `MILVUS_TIMEOUT` | `10` | Milvus 请求超时（秒） |

## 测试

```powershell
cd backend
.\.venv\Scripts\python.exe -m pytest
```

- 单元测试使用 `FakeEmbedder` / `InMemoryVectorStore` / `FakeChatModel`，不依赖外部服务；
- Milvus 集成测试会临时创建并删除测试 Collection。

## 常用 Docker 命令

```powershell
docker compose up -d        # 启动
docker compose ps           # 查看状态
docker compose stop         # 停止但保留数据
docker compose start        # 重新启动
docker compose logs -f milvus  # 查看 Milvus 日志
```

> 不要执行 `docker compose down -v`，它会删除数据卷，导致向量数据全部丢失。

## 常见问题

### Docker 报 WSL 更新失败

在管理员 PowerShell 中执行：

```powershell
wsl --install --no-distribution
```

必要时重启电脑，再重新启动 Docker Desktop。

### 镜像拉取失败

本项目 `docker-compose.yml` 已使用国内可访问镜像源。如切换到可访问官方源的环境，可将镜像名替换为：

```text
quay.io/coreos/etcd:v3.5.5
minio/minio:RELEASE.2023-03-20T20-16-18Z
milvusdb/milvus:v2.5.4
```

### Python 找不到 app 包

脚本需从 `backend` 目录运行；测试使用 `conftest.py` 自动处理导入路径。

## 安全说明

- 不要把真实 `.env` 提交到 Git；
- 不要在代码或日志中打印 API Key；
- 查看配置时请对 `*_API_KEY` 打码。

## License

仅用于学习与面试项目。
# 智答 Agent（document-intelligent-Q-A-Agent）

基于 **混合检索 + Agent 自主推理** 的文档智能问答系统。

> 目标不是复刻一个 PDF 问答 Demo，而是搭建一个具备生产级工程能力的 RAG 应用：
> 文档上传 → 解析入库 → 向量检索 → LLM 流式生成 → 引用溯源，并逐步加入混合检索、Rerank、Agent 工作流与评测体系。

## 当前进度

| 里程碑 | 内容 | 状态 |
|---|---|---|
| M0 | 工程骨架、配置、测试、VSCode 配置 | ✅ 已完成 |
| M1 | 文档解析、分块、Embedding、Milvus 入库与检索 | ✅ 已完成 |
| M2 | LLM 接入层、RAG 组装服务、引用溯源 | ✅ 已完成 |
| M3 | FastAPI 接口、SSE 流式输出、文档上传、原生前端 | ✅ 已完成 |
| M4 | 混合检索（向量 + BM25 + Rerank）与对比评测 | ⏳ 待开始 |
| M5 | LangGraph Agent 工作流（Router / Grader / Rewrite） | ⏳ 待开始 |
| M6 | LangSmith 追踪、检索评测、Docker 打包收尾 | ⏳ 待开始 |

## 技术栈

| 层级 | 技术 |
|---|---|
| 后端 | FastAPI + Uvicorn |
| Agent / LLM | OpenAI 兼容 SDK（可切换 DeepSeek / 百炼 / OpenAI 等） |
| 向量库 | Milvus 2.5（Docker Compose） |
| Embedding | OpenAI 兼容接口（示例：百炼 `qwen3.7-text-embedding`） |
| LLM | OpenAI 兼容接口（示例：`deepseek-v4-flash`、`qwen-plus`） |
| 前端 | 原生 HTML / CSS / JavaScript |
| 流式协议 | SSE（Server-Sent Events） |
| 测试 | pytest |
| 部署 | Docker Compose（Milvus + etcd + MinIO） |

## 核心特性

- 文档解析：支持 `.txt` / `.md` / `.pdf`
- 文本分块：支持 `chunk_size` 与 `chunk_overlap`
- 向量化入库：批量调用 Embedding API，写入 Milvus
- 语义检索：基于余弦相似度的 top-k 检索
- RAG 生成：检索结果编号注入 Prompt，LLM 依据资料回答
- 引用溯源：回答标注 `[1]`，返回 `filename`、`chunk_index`、`score`
- SSE 流式输出：模型 token 逐步推送，前端打字机效果
- 文件上传接口：浏览器直接上传文档并写入知识库
- 原生前端：对话、上传、流式渲染、引用卡片
- 可替换设计：`Embedder` / `VectorStore` / `ChatModel` 均面向接口编程

## 系统架构

```text
┌─────────────────────────────────────────────────────────┐
│                         浏览器                           │
│  聊天界面 / 文件上传 / ReadableStream 解析 SSE            │
└──────────────────────────┬──────────────────────────────┘
                           │ HTTP / SSE
┌──────────────────────────▼──────────────────────────────┐
│                        FastAPI                           │
│  POST /api/documents      文档上传与入库                  │
│  POST /api/chat           非流式问答                      │
│  POST /api/chat/stream    SSE 流式问答                    │
└──────────────────────────┬──────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────┐
│                        RAGService                        │
│  问题向量化 → Milvus 检索 → Prompt 组装 → LLM 生成        │
└──────────────────────────┬──────────────────────────────┘
                           │
        ┌──────────────────┼──────────────────┐
        ▼                  ▼                  ▼
   Embedding API       Milvus             LLM API
```

### Milvus 依赖组件

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
│   ├── main.py              # FastAPI 入口、静态页面托管
│   ├── config.py            # 配置定义与读取
│   ├── dependencies.py      # 依赖注入与单例服务
│   ├── schemas.py           # API 请求 / 响应模型
│   ├── api/routes/
│   │   ├── health.py        # 健康检查
│   │   ├── chat.py          # 问答与 SSE 流式接口
│   │   └── documents.py     # 文档上传接口
│   ├── ingestion/
│   │   ├── loader.py        # txt / md / pdf 解析
│   │   ├── splitter.py      # 文本分块
│   │   └── service.py       # 入库编排
│   ├── embedding/           # Embedding 接口与实现
│   ├── vectorstore/         # 向量库接口、内存版、Milvus 版
│   ├── llm/                 # ChatModel 接口、Fake、OpenAI 兼容实现
│   ├── rag/                 # RAGService 组装层
│   └── prompts/             # Prompt 模板
├── scripts/
│   ├── inspect_milvus.py    # 只读查看 Milvus 数据
│   ├── ingest_js_note.py    # 命令行文档入库演示
│   └── ask.py               # 命令行 RAG 问答
└── tests/                   # 单元测试与集成测试

frontend/
├── index.html               # 对话与上传页面
├── css/style.css            # 页面样式
└── js/app.js                # 上传、SSE 解析、流式渲染

docker-compose.yml           # Milvus + etcd + MinIO
.env.example                 # 环境变量模板
```

## 快速开始

### 1. 前置条件

- Python 3.11+
- Docker Desktop（Windows 使用 WSL2）

### 2. 安装依赖

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

编辑 `backend/.env`：

```text
EMBEDDING_BASE_URL=你的 Embedding 服务地址
EMBEDDING_API_KEY=你的 Embedding Key
EMBEDDING_MODEL=你的 Embedding 模型名

LLM_BASE_URL=你的 LLM 服务地址
LLM_API_KEY=你的 LLM Key
LLM_MODEL=你的 LLM 模型名
```

> `LLM_BASE_URL` / `LLM_API_KEY` 留空时，会复用 `EMBEDDING_BASE_URL` / `EMBEDDING_API_KEY`。
> `.env` 已被 `.gitignore` 忽略，请勿提交真实密钥。

### 4. 启动 Milvus

在项目根目录执行：

```powershell
docker compose up -d
docker compose ps
```

### 5. 启动后端

```powershell
cd backend
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload
```

访问：

- Web 界面：http://127.0.0.1:8000/
- 健康检查：http://127.0.0.1:8000/health
- Swagger 文档：http://127.0.0.1:8000/docs

### 6. 使用 Web 界面

1. 左侧选择 `.txt` / `.md` / `.pdf` 文件并上传；
2. 输入问题，点击发送；
3. 回答会通过 SSE 逐字显示；
4. 回答下方会展示引用来源卡片。

## API 说明

### 健康检查

```text
GET /health
```

### 文档上传

```text
POST /api/documents
Content-Type: multipart/form-data
字段：file
```

响应：

```json
{
  "document_id": "e16532bf63c722f5c3e7700dcfc08a3f",
  "filename": "js.md",
  "chunk_count": 31
}
```

### 非流式问答

```text
POST /api/chat
Content-Type: application/json

{
  "question": "const和let有什么区别",
  "top_k": 4
}
```

响应：

```json
{
  "answer": "...",
  "sources": [
    {
      "chunk_id": "...",
      "score": 0.8026,
      "text": "...",
      "metadata": {
        "filename": "js.md",
        "chunk_index": 0
      }
    }
  ]
}
```

### SSE 流式问答

```text
POST /api/chat/stream
Content-Type: application/json
```

事件协议：

```text
event: sources
data: {"sources":[...]}

event: delta
data: {"content":"根据"}

event: delta
data: {"content":"资料"}

event: done
data: {}

event: error
data: {"message":"..."}
```

前端不能使用 `EventSource`，因为该接口是 POST 请求。正确方式是：

```javascript
fetch("/api/chat/stream", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ question, top_k: 4 }),
})
```

然后通过 `response.body.getReader()` 逐块读取，并用 `TextDecoder` 处理 UTF-8。

## 命令行工具

### 文档入库

编辑 `backend/scripts/ingest_js_note.py` 的 `FILE_PATH` 后执行：

```powershell
cd backend
.\.venv\Scripts\python.exe scripts\ingest_js_note.py
```

### 命令行问答

```powershell
cd backend
.\.venv\Scripts\python.exe scripts\ask.py "你的问题" --top-k 3
```

### 查看 Milvus 数据

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

当前测试覆盖：

- 配置与健康检查
- 文档解析与分块
- Embedding 接口
- 内存向量库与 Milvus 集成
- RAGService 编排
- ChatModel 流式接口
- Chat / SSE / 文件上传 API

健康状态下预期：

```text
28 passed
```

## 常用 Docker 命令

```powershell
docker compose up -d          # 启动
docker compose ps             # 查看状态
docker compose stop           # 停止但保留数据
docker compose start          # 重新启动
docker compose logs -f milvus # 查看 Milvus 日志
```

> 不要执行 `docker compose down -v`，它会删除数据卷，导致向量数据全部丢失。

## 常见问题

### Docker Desktop 所有 API 返回 500

表现为：

```text
request returned 500 Internal Server Error for API route ... DockerDesktopLinuxEngine
```

通常是 Docker Desktop Linux Engine 假死，执行：

```powershell
docker desktop restart
docker compose up -d
```

### Docker 报 WSL 更新失败

在管理员 PowerShell 中执行：

```powershell
wsl --install --no-distribution
```

必要时重启电脑，再重新启动 Docker Desktop。

### 镜像拉取失败

本项目 `docker-compose.yml` 已使用国内可访问镜像源。如切换到可访问官方源的环境，可替换为：

```text
quay.io/coreos/etcd:v3.5.5
minio/minio:RELEASE.2023-03-20T20-16-18Z
milvusdb/milvus:v2.5.4
```

### Python 找不到 app 包

脚本需从 `backend` 目录运行；测试使用 `conftest.py` 自动处理导入路径。

### 前端页面没有样式或 JS 不加载

前端由 FastAPI 托管在：

```text
/static
```

`index.html` 中的资源路径应为：

```html
/static/css/style.css
/static/js/app.js
```

### 修改 CSS 后浏览器不更新

FastAPI 静态文件不会触发热更新。开发时请在 DevTools 中勾选：

```text
Network → Disable cache
```

然后使用 `Ctrl + Shift + R` 强制刷新。

### 使用 Live Server 时接口 404 / 跨域

Live Server 的 `5500` 端口和 FastAPI 的 `8000` 端口不同源。开发推荐直接访问：

```text
http://127.0.0.1:8000/
```

如果必须使用 Live Server，需要在后端配置 CORS，并让前端通过 `http://127.0.0.1:8000` 请求接口。

## 安全说明

- 不要把真实 `.env` 提交到 Git；
- 不要在代码或日志中打印 API Key；
- 查看配置时请对 `*_API_KEY` 打码；
- 前端渲染模型回答和文档内容时使用 `textContent`，避免 XSS。

## License

仅用于学习与面试项目。
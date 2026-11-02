# 智答 Agent（document-intelligent-Q-A-Agent）

基于 **混合检索 + 实体上下文注入 + 精排 + Agent 自主推理** 的文档智能问答系统。

> 目标不是复刻一个 PDF 问答 Demo，而是搭一套有数据质量意识的 RAG 工程：
> 文档上传 → 页级去噪 → 可插拔清洗 → 实体上下文 → 分块 → 向量化入库 → 混合检索 → 精排 → LLM 流式生成 → 引用溯源，
> 并且每一步改动都有可复现的评测数据支撑（含失败与负收益记录）。

## 当前进度

| 里程碑 | 内容 | 状态 |
|---|---|---|
| M0 | 工程骨架、配置、测试、VSCode 配置 | ✅ 已完成 |
| M1 | 文档解析、分块、Embedding、Milvus 入库与检索 | ✅ 已完成 |
| M2 | LLM 接入层、RAG 组装服务、引用溯源 | ✅ 已完成 |
| M3 | FastAPI 接口、SSE 流式输出、文档上传、原生前端 | ✅ 已完成 |
| M4 | 混合检索（向量 + BM25 + Rerank）与对比评测 | ✅ 已完成 |
| M5 | LangGraph Agent 工作流（Router / Grader / Rewrite） | ✅ 已完成 |
| M6 | 入库链路改造、知识库查看、追踪与交付收尾 | ✅ 已完成 |

M6 细分（详细过程见 `docs/召回质量问题清单.md`、`docs/M6-optimization-log.md`）：

| 子项 | 状态 | 说明 |
|---|---|---|
| 可插拔文档清洗管线 | ✅ | `app/ingestion/cleaning/`，5 个通用清洗器 + PDF 页级去噪 |
| 分块策略与对比实验 | ✅ | `fixed` / `recursive` 均可切，实测定稿 `fixed 500/50` |
| chunk 稳定锚点 id | ✅ | `doc_id#chunk_index:sha1[:8]`，重入库不变 |
| 重入库 + 内容锚点评测 | ✅ | 3 套内容锚点评测集，生产库 `qa_chunks_prod_v3` |
| 精排接入（百炼 text-rerank） | ✅ | 生产语料 Recall@5 0.9333 → 0.9667 |
| 知识库只读接口 | ✅ | `/api/knowledge/*` 三个接口 + 测试 |
| 知识库查看页面 | ✅ | `/knowledge`：文档列表 → chunk 列表 → 锚点高亮定位 |
| 来源卡片点击跳转 | ✅ | `sources` 事件带 `anchor_id` 等定位字段，卡片可点 |
| LangSmith 全链路追踪 | ✅ | 根 span + 4 个节点 + LLM span，单请求一条 trace |
| 应用本体 Docker 打包 | ✅ | `backend/Dockerfile`，compose 一键起 api + Milvus 依赖栈 |

## 技术栈

| 层级 | 技术 |
|---|---|
| 后端 | FastAPI + Uvicorn |
| Agent / LLM | OpenAI 兼容 SDK（可切换 DeepSeek / 百炼 / OpenAI 等） |
| Agent 工作流 | LangGraph 1.2 |
| 向量库 | Milvus 2.5（Docker Compose） |
| 检索 | 向量检索 + BM25 + RRF 融合 + Rerank 精排 |
| 精排 | 阿里云百炼 `qwen3.7-text-rerank`（text-rerank 协议） |
| Embedding | OpenAI 兼容接口（示例：`qwen3.7-text-embedding`，1024 维） |
| 分块 | `fixed` / `recursive`，默认 `fixed`，`chunk_size=500`、`chunk_overlap=50` |
| 前端 | 原生 HTML / CSS / JavaScript |
| 流式协议 | SSE（Server-Sent Events） |
| 可观测性 | LangSmith（默认关闭，按需开启；载荷脱敏） |
| 测试 | pytest（326 passed / 4 skipped） |
| 部署 | Docker Compose（api + Milvus + etcd + MinIO，一条命令） |

## 核心特性

**数据链路**

- 文档解析：`.txt` / `.md` / `.pdf`，PDF 保留页边界供页级去噪
- 页级去噪：识别并删除页眉、页脚、页码，双通道判定（全局比例阈值 + 连续页段）
- 可插拔清洗管线：`Normalize` → `Media` → `Noise` → `Whitespace` → `Dedup`，统一 `BaseCleaner.clean(text) -> str`
- 清洗可审计：保留每步 `cleaner_name`、字符数变化、清洗前后 diff，PDF 额外记录删除的页眉/页脚/页码键
- 实体上下文注入：每个块首加 `【极米科技(688696)｜2025年年度报告】`，Markdown 用最近章节标题
- 分块策略：`fixed` 与 `recursive` 可配置，按语料实测选定
- 稳定锚点 id：`{doc_id}#{chunk_index}:{sha1(text[:50])[:8]}`，同内容重入库 id 不变，hash 段可检测内容漂移
- 增量友好：同文档重入库先写新块再清理失效锚点，不整库重建

**检索**

- 向量检索：Milvus 余弦相似度 top-k
- BM25 关键词检索：jieba 分词 + rank_bm25
- 混合检索：向量与 BM25 通过 RRF 融合
- Rerank 精排：百炼 text-rerank（主用），保留 LLM Rerank 与 NoOp 基线实现
- 可配置检索模式：`vector` / `bm25` / `hybrid` / `hybrid_rerank`
- 精排可观测：失败自动回退原顺序，并记录 `fallback_count` 与最后一次错误原因
- 检索评测：Recall@k、Hit@k、MRR@k、nDCG@k 与延迟对比，支持内容锚点标注

**Agent 与生成**

- Router：自动区分 `direct` 直答与 `retrieve` 知识库检索
- Grader：过滤与问题无关的召回 chunk，降低上下文噪声
- Rewrite：检索无相关结果时改写查询，最多重试 `AGENT_MAX_REWRITES` 次
- 无依据拒答：多轮检索仍无相关 chunk 时明确告知资料不足，不让 LLM 自由编造
- RAG 生成：检索结果编号注入 Prompt，LLM 依据资料回答
- 引用溯源：回答标注 `[1]`，返回 `filename`、`chunk_index`、`score`
- SSE 流式输出：模型 token 逐步推送，前端打字机效果

**接口与前端**

- 文档上传接口：浏览器直接上传并写入知识库
- 知识库只读接口：文档列表、文档 chunk 列表、按锚点查单块
- 知识库页面：`/knowledge`，从对话页点击文档或来源卡片即可跳转定位
- 来源卡片可点击：携带 `anchor_id` 跳到知识库页面，滚动到对应块并高亮
- 原生前端：对话、上传、流式渲染、引用卡片
- 可替换设计：`Embedder` / `VectorStore` / `Retriever` / `Reranker` / `ChatModel` / `BaseCleaner` 均面向接口编程

## 系统架构

```text
┌─────────────────────────────────────────────────────────┐
│                         浏览器                           │
│  聊天界面 / 文件上传 / ReadableStream 解析 SSE            │
└──────────────────────────┬──────────────────────────────┘
                           │ HTTP / SSE
┌──────────────────────────▼──────────────────────────────┐
│                        FastAPI                           │
│  POST /api/documents          文档上传与入库              │
│  POST /api/chat               非流式问答                  │
│  POST /api/chat/stream        SSE 流式问答                │
│  GET  /api/knowledge/*        知识库只读查看              │
└──────────────────────────┬──────────────────────────────┘
                           │
        ┌──────────────────┴───────────────────┐
        ▼                                      ▼
┌───────────────────┐              ┌───────────────────────┐
│ IngestionService  │              │     AgentService      │
│ 解析→去噪→清洗    │              │ LangGraph 决策        │
│ →实体前缀→分块    │              │ → Prompt → LLM 流式   │
│ →向量化→入库      │              │ → 引用溯源            │
└─────────┬─────────┘              └───────────┬───────────┘
          │                                    │
          ▼                                    ▼
┌───────────────────┐              ┌───────────────────────┐
│ Embedding API     │              │  LangGraph Workflow   │
│ Milvus            │              │ Router→Retrieve→Grade │
└───────────────────┘              │   →Rewrite（可选）    │
                                   └───────────┬───────────┘
                                               ▼
                                   Embedding + Milvus / Rerank
```

### Agent 工作流

```text
START → Router
          ├─ direct   → 直接回答（不检索、不返回引用）
          └─ retrieve → Retrieve → Grader
                                  ├─ 有相关 chunk → 生成答案并返回引用
                                  └─ 无相关 chunk → Rewrite → Retrieve（最多重试）
                                                    └─ 仍无结果 → 明确告知资料不足
```

- `Router`：判断是否需要检索知识库；无法确定时保守选择 `retrieve`。
- `Retriever`：复用 M4 的向量 / BM25 / Hybrid / Rerank 检索链路。
- `Grader`：判断候选 chunk 是否与问题相关，只把相关结果交给生成层。
- `Rewrite`：检索失败时生成新查询，最多重试 `AGENT_MAX_REWRITES` 次。
- **异常回退**：Router 输出无效时使用规则路由；Grader 输出无效时保留全部候选；Rewrite 输出无效时使用保守改写。
- **流式兼容**：路由、检索和评分在生成前完成，最终答案仍通过 SSE `delta` 事件逐 token 输出。

### Milvus 依赖组件

```text
Milvus 容器
├── etcd     保存元数据
└── MinIO    保存向量数据与索引文件
```

- `19530`：pymilvus 连接端口
- `9091`：Milvus 健康检查 / 指标
- `9001`：MinIO 管理控制台

## 数据链路：从文件到 chunk

```text
文件
 └─ parse_document            保留 PDF 页边界
     └─ PdfPageNoiseCleaner   PDF：删页眉/页脚/页码（全局阈值 + 连续页段）
         └─ dewrap + join     跨页断行合并、页与页拼接
             └─ CleanerPipeline  通用清洗：Normalize → Media → Noise → Whitespace → Dedup
                 └─ build_chunk_header  块首注入实体上下文
                     └─ split_text      fixed / recursive
                         └─ embed_batch 批量向量化
                             └─ Milvus upsert（锚点 id 为主键）
```

### 清洗器清单

| 清洗器 | 职责 |
|---|---|
| `PdfPageNoiseCleaner` | 页眉、页脚、页码；双通道判定，支持分段排版的多套页眉 |
| `NormalizeCleaner` | 全角→半角、去 BOM、统一换行、去零宽字符 |
| `MediaCleaner` | Markdown 图片、base64 长串 → 占位或丢弃 |
| `NoiseCleaner` | 去水印行、孤立页码等规则噪声 |
| `WhitespaceCleaner` | 合并连续空行 / 多空格、行首行尾 strip |
| `DedupCleaner` | 去完全重复段落（扫描件、PDF 常见） |

页级去噪为什么需要双通道：成都银行年报 267 页分两段排版，两套页眉各占 45% / 44%，
都过不了 `min_ratio=0.6` 的全局阈值，导致 121 处页眉被跨页合并粘进正文（形如 `2025年年度报告59`）。
补上 `min_run=8` 的连续页段识别后残留归零。实测该文档删除 533 行 / 14,379 字（占原文 6%），正文抽查完整。

### 实体上下文为什么是最大的杠杆

年报里"主要会计数据""研发投入"等表格在多家公司之间措辞高度同构，而块正文通常不含公司名，
实体只出现在 query 里——问 A 公司会返回 B 公司的表。同一套块边界、同一套融合参数，
只给每块加一行 `【公司(代码)｜年份】`，pdf5 数据集上 hybrid Recall@5 从 0.54 提到 0.82，vector 从 0.24 提到 0.72。

前缀同时进入 BM25 与 embedding；锚点 id 仍基于**原始块**计算，保证开关上下文前后 id 稳定可比。

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
│   │   ├── documents.py     # 文档上传接口
│   │   └── knowledge.py     # 知识库只读查看接口
│   ├── ingestion/
│   │   ├── loader.py        # txt / md / pdf 解析
│   │   ├── pdf_text.py      # PDF 页级去噪后的 dewrap 与拼接
│   │   ├── cleaning/        # 可插拔清洗管线
│   │   │   ├── base.py      # BaseCleaner 接口
│   │   │   ├── pipeline.py  # 串行管线与 trace
│   │   │   ├── normalize.py / whitespace.py / dedup.py
│   │   │   ├── noise.py / media.py / code_regions.py
│   │   │   ├── pdf_page_noise.py  # 页眉页脚页码
│   │   │   ├── report.py    # 清洗报告与 diff
│   │   │   └── factory.py   # 默认管线装配
│   │   ├── context.py       # 实体上下文前缀 / 最近章节标题
│   │   ├── splitter.py      # fixed / recursive 分块
│   │   ├── manifest.py      # 入库清单
│   │   └── service.py       # 入库编排、锚点 id、增量替换
│   ├── embedding/           # Embedding 接口与实现
│   ├── vectorstore/         # 向量库接口、内存版、Milvus 版
│   ├── llm/                 # ChatModel 接口、Fake、OpenAI 兼容实现
│   ├── agent/               # LangGraph Router / Grader / Rewrite / Service
│   ├── rag/                 # RAGService 组装层
│   ├── retrieval/           # 向量、BM25、RRF、Rerank 检索器
│   ├── rerank/              # DashScope / LLM / NoOp Reranker
│   ├── observability/       # LangSmith 追踪封装（关闭时零开销）
│   ├── evaluation/          # Recall、MRR、nDCG 指标与评测运行器
│   └── prompts/             # Prompt 模板
├── scripts/
│   ├── ingest_corpus.py     # 批量语料入库 + manifest
│   ├── ingest_js_note.py    # 单文档入库演示
│   ├── preview_cleaning.py  # 清洗前后 diff 预览
│   ├── ask.py               # 命令行 RAG 问答
│   ├── inspect_milvus.py    # 只读查看 Milvus 数据
│   ├── evaluate_retrieval.py    # 多模式检索评测
│   ├── validate_content_eval.py # 校验评测集 snippet 是否存在于语料
│   ├── tune_fusion.py       # RRF 融合参数离线网格扫描
│   ├── build_ctx_collection.py  # 给存量库补实体前缀
│   └── probe_*.py           # 精排策略探针（MaxP / 融合权重）
├── evaluation/
│   ├── retrieval_cases_content.jsonl  # 45 题，生产语料（20 Vue + 25 年报）
│   ├── retrieval_cases_pdf.jsonl      # 25 题，5 份年报
│   ├── retrieval_cases_pilot.jsonl    # 14 题，早期 pilot
│   ├── qa_refusal_cases.jsonl         # 拒答样本
│   ├── corpus_manifest.json           # 最近一次入库清单
│   └── results_*.json                 # 各次评测的完整逐题结果
└── tests/                   # 单元测试与集成测试

frontend/
├── index.html               # 对话与上传页面
├── knowledge.html           # 知识库只读查看页面
├── css/style.css            # 对话页样式
├── css/knowledge.css        # 知识库页样式
├── js/app.js                # 上传、SSE 解析、流式渲染、来源卡片跳转
└── js/knowledge.js          # 文档列表、chunk 列表、锚点定位

docs/
├── M6-optimization-log.md   # M6 基线冻结记录
└── 召回质量问题清单.md      # 召回问题的定位、实验、结论与负结果

docker-compose.yml           # api + Milvus + etcd + MinIO
backend/Dockerfile           # 后端镜像（依赖分层、非 root、自带 healthcheck）
.dockerignore                # 构建上下文排除 venv / 密钥 / 评测数据
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

RERANK_API_URL=https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank
RERANK_API_KEY=你的百炼 Key
RERANK_MODEL=qwen3.7-text-rerank

RETRIEVAL_MODE=hybrid_rerank
RETRIEVAL_CANDIDATE_K=50
MILVUS_COLLECTION_NAME=qa_chunks_prod_v3
CHUNK_STRATEGY=fixed
CHUNK_SIZE=500
CHUNK_OVERLAP=50
AGENT_MAX_REWRITES=1
```

> `LLM_BASE_URL` / `LLM_API_KEY` 留空时，会复用 `EMBEDDING_BASE_URL` / `EMBEDDING_API_KEY`。
> `.env` 已被 `.gitignore` 忽略，请勿提交真实密钥。

### 4. 启动（二选一）

**方式一：Docker 一键启动（推荐，验收用这个）**

```powershell
docker compose up -d --build
docker compose ps
```

首次构建约 3–5 分钟；之后只改 `backend/app` 或 `frontend` 时重建很快（依赖单独一层，不重装）。
`api` 容器自带 healthcheck，`docker compose ps` 里显示 `healthy` 即就绪。

两个容易踩的点：

- 密钥通过 `env_file: ./backend/.env` 注入，**不会打进镜像**（`.dockerignore` 已排除）；
- 容器里的 `127.0.0.1` 指容器自己，所以 compose 把 `MILVUS_URI` 覆盖成 `http://milvus:19530`。
  若日后把 Embedding / LLM / Rerank 换成本机服务，同样要改成 `http://host.docker.internal:端口`。

**方式二：本地开发（改代码即时生效）**

```powershell
docker compose up -d etcd minio milvus   # 只起依赖栈
cd backend
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload
```

两种方式访问地址相同：

- Web 界面：http://127.0.0.1:8000/
- 知识库页：http://127.0.0.1:8000/knowledge
- 健康检查：http://127.0.0.1:8000/health
- Swagger 文档：http://127.0.0.1:8000/docs

> 端口冲突：Docker 版与本地版都占用 8000，切换时先停掉另一个。

### 5. 使用 Web 界面

1. 左侧选择 `.txt` / `.md` / `.pdf` 文件并上传（自动走清洗、实体前缀、分块、入库）；
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
      "chunk_id": "e16532bf...#0:8f3a2c1b",
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

### 知识库只读接口

```text
GET /api/knowledge/documents
```

返回文档列表，含 `document_id`、`doc_name`、`chunk_count`、分块参数与 embedding 模型 / 维度。

```text
GET /api/knowledge/documents/{document_id}/chunks
```

返回该文档的全部 chunk：`anchor_id`、`chunk_index`、`page_number`、`block_type`、`text`、`chunk_hash`。

```text
GET /api/knowledge/chunks?anchor_id={anchor_id}
```

按锚点 id 定位单个 chunk，供引用卡片跳转使用。

### 知识库页面

```text
GET /knowledge
```

只读页面，路由用 `location.hash` 承载：

```text
/knowledge#/knowledge?doc={document_id}&anchor={anchor_id}
```

- 只带 `doc`：打开该文档的 chunk 列表；
- 只带 `anchor`：先反查锚点所属文档再定位；
- 命中锚点时滚动到对应块并高亮；锚点失效（文档已重建）时给出提示。

对话页的来源卡片和左侧文档列表都会生成这类链接。

## 命令行工具

| 脚本 | 用途 |
|---|---|
| `ingest_corpus.py` | 批量入库并生成 manifest，支持 `--docs` / `--chunk-*` / `--context-header` / `--dry-run` |
| `ingest_js_note.py` | 单文档入库演示 |
| `preview_cleaning.py` | 打印清洗前后 diff 与报告，调规则时先用它 |
| `ask.py` | 命令行 RAG 问答 |
| `inspect_milvus.py` | 只读查看 Milvus 中的块 |
| `evaluate_retrieval.py` | 四模式检索评测，支持 `--dataset` / `--modes` / `--candidate-k` / `--show-misses` |
| `validate_content_eval.py` | 校验评测集 snippet 是否真实存在于清洗后语料 |
| `tune_fusion.py` | 候选只取一次，离线穷举 `rrf_k × 权重` |
| `build_ctx_collection.py` | 给存量 collection 补实体前缀，不重新解析 |

```powershell
cd backend
# 批量入库（16 篇生产语料）
.\.venv\Scripts\python.exe scripts\ingest_corpus.py `
  --corpus-root "<语料根目录>" --collection qa_chunks_prod_v3 `
  --strategy fixed --chunk-size 500 --chunk-overlap 50 --docs <相对路径...>

# 清洗预览（调规则时先看 diff）
.\.venv\Scripts\python.exe scripts\preview_cleaning.py "path\to\file.pdf"

# 命令行问答
.\.venv\Scripts\python.exe scripts\ask.py "你的问题" --top-k 3

# 查看 Milvus 数据
.\.venv\Scripts\python.exe scripts\inspect_milvus.py --limit 10
```

## 检索评测

评测集位于 `backend/evaluation/`，全部使用**内容锚点**标注：

```json
{ "id": "jimo-001", "question": "...", "relevant_doc": "上市公司年报/极米科技_2025年年度报告.pdf",
  "relevant_snippets": ["2025 年,极米科技实现营业收入34.67 亿元"], "category": "...", "difficulty": "..." }
```

命中判定用"检索结果文本是否包含 snippet"，因此改分块参数后**评测集不需要重标**。

| 评测集 | 题数 | 构成 |
|---|---:|---|
| `retrieval_cases_content.jsonl` | 45 | 生产语料：20 题 Vue 3 文档 + 25 题 5 份上市公司年报 |
| `retrieval_cases_pdf.jsonl` | 25 | 仅年报，考数值 / 表格类问题 |
| `retrieval_cases_pilot.jsonl` | 14 | 早期 pilot：4 题技术文档 + 10 题年报 |

生产索引 `qa_chunks_prod_v3`：16 篇文档 / 2620 块 / 1024 维 / `fixed 500/50` / 全量块带实体前缀。

### 四模式对比（content 45 题，`top_k=5`，`candidate_k=50`）

| 模式 | Recall@5 | Hit@5 | MRR@5 | nDCG@5 | 平均延迟 |
|---|---:|---:|---:|---:|---:|
| `vector` | 0.8222 | 0.8222 | 0.6415 | 0.7110 | 247.45 ms |
| `bm25` | 0.8000 | 0.8000 | 0.6589 | 0.7220 | 16.72 ms |
| `hybrid` | 0.9333 | 0.9333 | 0.7619 | 0.8331 | 192.32 ms |
| `hybrid_rerank` | **0.9667** | **0.9778** | **0.7981** | **0.8831** | 435.04 ms |

> 延迟为 2026-11-02 Milvus 读路径统一降为 `Bounded` 一致性后的重测值（此前 vector 461.67 / hybrid 482.01 / hybrid_rerank 727.25 ms）。
> 一致性级别只影响等待时间，不影响召回，四项质量指标与重测前逐位一致。

完整逐题结果见 `backend/evaluation/results_content_v3_4modes.json`。

### 三套评测集上的当前成绩

| 评测集 | hybrid Recall@5 | hybrid_rerank Recall@5 | 精排 MRR@5 | 精排 nDCG@5 |
|---|---:|---:|---:|---:|
| content 45 | 0.9333 | **0.9667** | 0.7981 | 0.8831 |
| pdf 25 | 0.8800 | **0.9400** | 0.7033 | 0.8139 |
| pilot 14 | 0.9286 | **0.9643** | 0.7143 | 0.8044 |

三套数据 `rerank_fallback_count` 均为 0，说明精排链路没有静默降级。

### 优化过程（同一套评测集上的演进）

| 阶段 | content 45 hybrid | pdf 25 hybrid | pilot 14 hybrid |
|---|---:|---:|---:|
| M4 基线（`qa_chunks_prod_v1`，无上下文） | 0.700 | 0.52 | 0.607 |
| + 实体上下文（`prod_v2`） | 0.922 | 0.86 | 0.821 |
| + 页眉去噪（`prod_v3`） | 0.9333 | 0.88 | 0.9286 |
| + 百炼精排 | **0.9667** | **0.9400** | **0.9643** |

> 基线行取自 `candidate_k=10` 时期（当时尚未带实体上下文）。扩池到 50 后单路与 hybrid 还会有小幅变化，
> 逐项对比见 `docs/召回质量问题清单.md` 第 7.2 节。
>
> 延迟受网络、模型服务和运行环境影响，只适合同一环境内的相对比较。本轮重测的精排增量为 **243 ms**
>（435.04 − 192.32；此前在不同环境下测得 250~550 ms），精排请求自身中位数 254 ms（`RERANK_MAX_CANDIDATES=30`）。

### 验证过但不采用的方案

负结果同样记录在 `docs/召回质量问题清单.md`，避免重复踩坑：

- **`chunk_overlap` 50 → 80**：块数 +7%，但 content 45 hybrid 掉到 0.800、pdf 25 掉到 0.640。财报表格区数字密集，多带的尾巴会把上一块的数字串拖进块首，向量和 BM25 排序一起变差。已回滚。
- **表格感知解析（pdfplumber `find_tables`）**：254 页年报前 10 页就要 4.5 s，整文档 900 s 未跑完，且表格块会绕开页眉去噪与实体前缀这条收益最大的链路；覆盖面只有 1/45 题。实现已删除。
- **句子级 MaxP 精排**：把只含答案片段的噪声块顶上来了，pdf 25 掉到 0.600。
- **去掉块首实体前缀**：精排 MRR 0.638 → 0.366，说明前缀对精排是正作用。
- **800 字块**：pdf 25 掉到 0.76，"PDF 建议 800 字"的经验在这套语料上不成立。

### 复现

```powershell
cd backend
# 四模式对比
.\.venv\Scripts\python.exe scripts\evaluate_retrieval.py `
  --dataset evaluation\retrieval_cases_content.jsonl `
  --candidate-k 50 --top-k 5 `
  --modes vector bm25 hybrid hybrid_rerank `
  --show-misses --output evaluation\results_content_v3_4modes.json

# 只看精排（快）
.\.venv\Scripts\python.exe scripts\evaluate_retrieval.py `
  --dataset evaluation\retrieval_cases_pdf.jsonl `
  --candidate-k 50 --top-k 5 --modes hybrid hybrid_rerank
```

## 可观测性：LangSmith 全链路追踪

**默认关闭**：`LANGSMITH_TRACING=false`（或没填 key）时不创建客户端、不发任何网络请求。
`tests/conftest.py` 里用环境变量强制关闭，避免开发者本地开了追踪后单测往远端刷几百条 trace。

开启方式（写进 `backend/.env`）：

```text
LANGSMITH_API_KEY=lsv2_pt_xxx
LANGSMITH_TRACING=true
LANGSMITH_PROJECT=smart-qa-agent
```

一次问答对应一条 trace，`/api/chat` 与 `/api/chat/stream` 结构一致：

| span | 类型 | 记录内容 |
|---|---|---|
| `AgentService.answer` | chain | 根 span：request_id、检索模式、candidate_k、collection、来源数与回答字数 |
| `LangGraph` | chain | 图执行整体耗时 |
| `router` / `retrieve` / `grade` / `rewrite` | chain | LangGraph 原生上报，节点名即 run_name |
| `_route_after_router` / `_route_after_grader` | chain | 条件边走向（direct / retrieve / ready / rewrite） |
| `generate` | llm | 首 token 延迟、总字数、耗时；输入是完整 system + user 消息 |

三个实现细节，都是实测撞出来的：

1. **节点 span 不手动埋点**。LangGraph 在追踪开启时会自动为每个节点上报，再手动加一层会出现同名 span 重复，且两层都会上传完整 state。
2. **载荷必须脱敏**。原生节点输出是整个 AgentState，含 50 条候选正文；客户端挂 `hide_inputs` / `hide_outputs` 把候选压成 `{id, score}`，单条 trace 的输出体积从几十 KB 降到约 1.8 KB。
3. **SSE 端点必须是 async**。同步生成器会被 Starlette 丢进线程池迭代，`with` 的进入与退出跨线程，根 span 上报不出去（表现为只剩一个孤立的 `generate`）。改成 async 端点后，阻塞的检索与打分再用 `run_in_threadpool` 隔离，不卡事件循环。

关掉追踪时，这些埋点全部走 `None` 分支，链路行为与未接入前完全一致。

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
| `MILVUS_COLLECTION_NAME` | `qa_chunks_prod_v3` | 当前生产 Collection |
| `MILVUS_TIMEOUT` | `10` | Milvus 请求超时（秒） |
| `RERANK_API_URL` | 百炼 text-rerank 地址 | Rerank 接口地址 |
| `RERANK_API_KEY` | 空 | Rerank API Key（可与 Embedding 共用） |
| `RERANK_MODEL` | `qwen3.7-text-rerank` | Rerank 模型名 |
| `RERANK_TIMEOUT` | `30` | Rerank 请求超时（秒） |
| `RERANK_MAX_CANDIDATES` | `30` | 送精排的候选数（实测 30 已饱和） |
| `RETRIEVAL_MODE` | `hybrid_rerank` | `vector` / `bm25` / `hybrid` / `hybrid_rerank` |
| `RETRIEVAL_CANDIDATE_K` | `50` | 混合检索与精排前的候选数量 |
| `CHUNK_STRATEGY` | `fixed` | `fixed` / `recursive` |
| `CHUNK_SIZE` | `500` | 分块长度 |
| `CHUNK_OVERLAP` | `50` | 分块重叠（不建议调大，见负结果） |
| `AGENT_MAX_REWRITES` | `1` | Grader 无相关结果时的最大查询改写次数 |
| `LANGSMITH_TRACING` | `false` | 是否开启 LangSmith 追踪，需同时配置 key |
| `LANGSMITH_API_KEY` | 空 | LangSmith API Key，空则追踪不生效 |
| `LANGSMITH_PROJECT` | `smart-qa-agent` | LangSmith 项目名（trace 归档位置） |
| `LANGSMITH_ENDPOINT` | `https://api.smith.langchain.com` | 上报地址，国内可换 agent 网关 |

## 测试

```powershell
cd backend
.\.venv\Scripts\python.exe -m pytest
```

当前 326 passed / 4 skipped（skip 为需要外部服务的用例）。覆盖范围：

- 配置与健康检查
- 文档解析、PDF 页边界、分块（fixed / recursive）
- 清洗管线：各清洗器、管线 trace、清洗报告、年报清洗回归、PDF 页眉页码去噪
- 实体上下文前缀与锚点 id 稳定性
- Embedding 接口、内存向量库与 Milvus 集成
- BM25、RRF 混合检索、精排（DashScope / LLM / NoOp）与回退逻辑
- 检索评测指标、内容锚点数据集加载
- LangGraph Router / Grader / Rewrite 分支与异常回退
- Agent 直答、RAG 生成、无依据拒答与流式输出
- Chat / SSE / 文件上传 / 知识库接口
- 知识库页面与静态资源路由
- `SourceOut` 定位字段组装与回退
- 入库 manifest 与上下文注入
- LangSmith 追踪封装：默认关闭不联网、载荷脱敏、span 参数装配

## 已知问题与下一步

1. **块边界问题未解**：`tongwei-001` 的答案行被 500 字窗口从中间切断，top5 第 5 条是同文档邻居块。加 overlap 已验证是负收益，方向改为「检索后把命中块的 ±1 邻居一并作为上下文」。
2. **多事实问题**：`iflytek-005` 两条 snippet 分属两块，需要查询拆解，与分块无关。
3. **回答中的 Markdown 未渲染**：前端用 `textContent` 输出（防 XSS），`**加粗**` 等标记会原样显示；如需渲染应引入白名单式渲染器。
4. **`/documents/{id}/chunks` 仍需全表扫描**：`metadata_json` 是整块 VARCHAR，无法按 `document_id` 在服务端过滤，单次约 300ms。根治要给 collection 加独立标量字段并重灌数据，当前量级可接受。
5. **chunk metadata 载荷偏大**：平均每块约 1.1KB（与正文相当），`page_cleaning` 等调试字段占了不少；精简后可再降全表扫描时间。

> Milvus 一致性已统一：`search()` / `all_chunks()` / `get_by_ids()` 全部为 `Bounded`（写入侧 `upsert`/`delete` 同步 `flush`，新数据立即可见）。实测向量检索单次 397ms → 9ms、知识库首屏约 1.1s → 0.33s，基准脚本见 `backend/scripts/bench_all_chunks.py`。

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

`backend/Dockerfile` 的基础镜像同理固定为华为云镜像源（`swr.cn-north-4.myhuaweicloud.com/ddn-k8s/docker.io/library/python:3.11-slim`）。
**如果你机器的 Docker 配了 registry 加速器**（Docker Desktop → Settings → Docker Engine → `registry-mirrors`），
加速器证书失效时会报 `x509: certificate is not valid for any names`——这类报错来自加速器，与项目无关，
要么在加速器里换一个可用地址，要么直接把 Dockerfile 的 `FROM` 换成能拉通的完整镜像地址。

### 容器启动即退出 / healthcheck 一直 starting

先看日志，不要猜：

```powershell
docker logs smart-qa-api --tail 50
```

两个真实踩过的案例：

- `RuntimeError: Form data requires "python-multipart" to be installed`：上传接口的路由注册依赖它，
  但本地 venv 里恰好装了、`requirements.txt` 漏了。已补入依赖清单。
- `failed to read .env: line 1: unexpected character "#" in variable name`：`backend/.env` 首行混入了
  非注释文本。pydantic 会忽略这类行，但 compose 的 `env_file` 解析器更严格，直接拒绝启动。

### 精排好像没生效

先看评测报告里的 `diagnostics.rerank_fallback_count`：

- `0`：精排正常；
- 大于 0：精排接口异常并已静默回退原顺序，报告里同时有 `last_fallback_error`。

历史踩坑：某 rerank 服务在本机走代理必然 502、直连超时，重试无意义。**先查服务商 `/models` 确认真实模型名，再写适配器。**

### Python 找不到 app 包

脚本需从 `backend` 目录运行；测试使用 `conftest.py` 自动处理导入路径。

### 前端页面没有样式或 JS 不加载

前端由 FastAPI 托管在 `/static`，`index.html` 中的资源路径应为：

```html
/static/css/style.css
/static/js/app.js
```

### 修改 CSS 后浏览器不更新

FastAPI 静态文件不会触发热更新。开发时请在 DevTools 中勾选 `Network → Disable cache`，
然后使用 `Ctrl + Shift + R` 强制刷新。

### 使用 Live Server 时接口 404 / 跨域

Live Server 的 `5500` 端口和 FastAPI 的 `8000` 端口不同源。开发推荐直接访问 `http://127.0.0.1:8000/`。

### 重入库会丢数据吗

不会。`IngestionService` 先写入新锚点，再删除同文档中已不存在的旧锚点，属于安全替换。
但**不要执行 `docker compose down -v`**，它会删除数据卷，向量数据全部丢失。

## 安全说明

- 不要把真实 `.env` 提交到 Git；
- 不要在代码或日志中打印 API Key；
- 查看配置时请对 `*_API_KEY` 打码；
- 前端渲染模型回答和文档内容时使用 `textContent`，避免 XSS。

## License

仅用于学习与面试项目。

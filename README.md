# 智答 Agent（document-intelligent-Q-A-Agent）

基于混合检索 + Agent 自主推理的文档智能问答系统。

> 当前阶段：M0 工程基础已就绪。后续里程碑：文档入库 → 基础 RAG → 混合检索 + Rerank → LangGraph Agent 工作流 → 引用溯源与评测 → Docker 部署。

## 快速开始（当前 M0）

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --reload
```

启动后访问：

- 健康检查：http://127.0.0.1:8000/health
- 接口文档：http://127.0.0.1:8000/docs

## 运行测试

```powershell
cd backend
.\.venv\Scripts\python.exe -m pytest
```
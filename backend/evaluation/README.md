# 检索评测集

`retrieval_cases.jsonl` 用于比较不同检索策略的效果，不调用 LLM 生成回答。

## 标注格式

每行是一个 JSON 对象：

```json
{
  "id": "js-001",
  "question": "var 声明的变量有哪些特点？",
  "relevant": [
    {"filename": "js.md", "chunk_index": 0}
  ]
}
```

- `id`：样本唯一标识。
- `question`：用户问题。
- `relevant`：回答该问题所需的相关 chunk。
- `filename` + `chunk_index` 与入库元数据对应，比 Milvus 主键更易读。

当前标注基于 `js.md` 按项目默认 `chunk_size` / `chunk_overlap` 切分后的 31 个 chunk。
如果修改了分块参数或重新生成文档，需要同步重新标注。

## 指标

- `Recall@k`：top-k 中命中的相关 chunk 占全部相关 chunk 的比例。
- `Hit@k`：top-k 中是否至少命中一个相关 chunk。
- `MRR@k`：第一个相关 chunk 的倒数排名。
- `nDCG@k`：二值相关性下的归一化折损累计增益。
- `latency`：单条问题的平均检索耗时，包含 Embedding / Rerank 网络请求。

## 运行

```powershell
cd backend
.\.venv\Scripts\python.exe scripts\evaluate_retrieval.py
```

指定模式或保存完整结果：

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_retrieval.py `
  --modes vector bm25 hybrid hybrid_rerank `
  --top-k 5 --candidate-k 10 `
  --show-misses --output evaluation\results.json
```
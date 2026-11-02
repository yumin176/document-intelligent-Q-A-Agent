# 检索评测集

评测只比较检索质量，不调用 LLM 生成回答。

## 文件清单

| 文件 | 题数 | 构成 | 用途 |
|---|---:|---|---|
| `retrieval_cases_content.jsonl` | 45 | 20 题 Vue 3 技术文档 + 25 题上市公司年报（5 家） | 生产语料主评测集，最贴近线上 |
| `retrieval_cases_pdf.jsonl` | 25 | 仅上市公司年报 | 数值 / 表格类问题专项 |
| `retrieval_cases_pilot.jsonl` | 14 | 4 题技术文档 + 10 题年报 | 早期 pilot，做交叉验证 |
| `qa_refusal_cases.jsonl` | 3 | 知识库外问题（未来股价 / 未来市值 / Vue 4 文档） | 拒答链路，不计入 Recall / MRR / nDCG |
| `retrieval_cases.jsonl` | 27 | 单篇 `js.md`，用 `filename + chunk_index` 标注 | 迁移前旧格式，仅作对照 |

## 标注格式（内容锚点，当前标准）

```json
{
  "id": "jimo-001",
  "question": "极米科技 2025 年营业收入是多少？",
  "relevant_doc": "上市公司年报/极米科技_2025年年度报告.pdf",
  "relevant_snippets": ["2025 年,极米科技实现营业收入34.67 亿元"],
  "category": "数值抽取",
  "difficulty": "easy"
}
```

- `relevant_doc`：语料库相对路径，和入库时的 `doc_name` 一致。
- `relevant_snippets`：回答该问题所依据的原文片段，命中判定为「检索结果文本包含该片段」。
- 一条问题可以有多个 snippet（多事实题），此时 Recall 按覆盖比例算。

**为什么用内容锚点**：分块策略、`chunk_size`、`chunk_overlap` 改动后，`chunk_index` 全变，
而 snippet 只跟原文有关。换分块参数不需要重标评测集，这是能连续做分块实验的前提。

旧的 `retrieval_cases.jsonl` 使用 `filename + chunk_index`，一旦改分块参数就必须重新标注，
保留它只为对照迁移前后的口径差异。

## 指标

- `Recall@k`：top-k 中命中的相关片段占全部相关片段的比例。
- `Hit@k`：top-k 中是否至少命中一个相关片段。
- `MRR@k`：第一个相关结果的倒数排名。
- `nDCG@k`：二值相关性下的归一化折损累计增益。
- `latency`：单条问题的平均检索耗时，包含 Embedding / Rerank 网络请求。
- `diagnostics.rerank_fallback_count`：精排失败并静默回退的次数，正常应为 0。

## 当前成绩

生产索引 `qa_chunks_prod_v3`（16 篇 / 2620 块 / 1024 维 / `fixed 500/50`），`top_k=5`、`candidate_k=50`：

| 评测集 | hybrid Recall@5 | hybrid_rerank Recall@5 |
|---|---:|---:|
| content 45 | 0.9333 | **0.9667** |
| pdf 25 | 0.8800 | **0.9400** |
| pilot 14 | 0.9286 | **0.9643** |

完整逐题结果见 `results_content_v3_4modes.json` 等文件；四模式对比见根目录 README。

## 运行

```powershell
cd backend
.\.venv\Scripts\python.exe scripts\evaluate_retrieval.py
```

指定数据集、模式或保存完整结果：

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_retrieval.py `
  --dataset evaluation\retrieval_cases_content.jsonl `
  --modes vector bm25 hybrid hybrid_rerank `
  --top-k 5 --candidate-k 50 `
  --show-misses --output evaluation\results_content_v3_4modes.json
```

## 标注校验

新增或修改 snippet 后，先确认它真的存在于清洗后的语料里，否则会得到假的 miss：

```powershell
.\.venv\Scripts\python.exe scripts\validate_content_eval.py `
  --dataset evaluation\retrieval_cases_content.jsonl `
  --corpus-root "<语料根目录>"
```

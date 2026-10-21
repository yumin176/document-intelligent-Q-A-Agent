"""Prompt 模板集中管理。"""


def build_rag_system_prompt(context: str) -> str:
    """根据检索到的上下文构造 system prompt。"""
    return (
        "你是一名严谨的知识库问答助手。\n"
        "请只依据下面【资料】回答用户问题；资料中没有的内容不要编造。\n"
        "如果资料不足以回答，请明确说明“资料中未找到相关信息”。\n"
        "在回答中引用资料时，请在相关句子末尾标注编号，例如 [1]、[2]。\n\n"
        f"【资料】\n{context}"
    )
def build_rerank_prompt(query: str, candidates: list) -> str:
    # 先处理候选数据，格式化成清晰的文本，便于llm理解
    blocks = []
    for index, candidate in enumerate(candidates, start=1):
        blocks.append(
            f"[{index}] id={candidate.id}\n"
            f"text={candidate.text[:500]}"
        )

    candidate_text = "\n\n".join(blocks)

    return (
        "你是检索结果重排序器。请根据用户问题，对候选文档按相关性打分，范围 0-10。\n"
        "只输出 JSON 数组，不要输出 Markdown，不要解释。\n"
        '每一项格式：{"id": "候选id", "score": 0-10}\n'
        "只能使用给定的 id，不要编造。\n\n"
        f"用户问题：\n{query}\n\n"
        f"候选文档：\n{candidate_text}"
    )

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


def build_router_prompt(question: str) -> str:
    """构造 Router 的路由判断 Prompt。"""
    return (
        "你是文档问答系统的路由器。请判断用户问题是否需要检索知识库。\n"
        "只输出 JSON，不要输出 Markdown、解释或额外文字。\n"
        '输出格式：{"route":"retrieve"} 或 {"route":"direct"}\n\n'
        "选择 direct：问候、感谢、闲聊、询问助手身份或能力等不需要资料的问题。\n"
        "选择 retrieve：涉及事实、概念、文档内容、代码知识或需要依据资料回答的问题。\n"
        "无法确定时选择 retrieve。\n\n"
        f"用户问题：\n{question}"
    )

def build_grader_prompt(question: str, candidates: list) -> str:
    """构造候选 chunk 相关性判断 Prompt。"""
    blocks = []
    for index, candidate in enumerate(candidates, start=1):
        source_name = (
            candidate.metadata.get("filename")
            or candidate.metadata.get("source")
            or candidate.id
        )
        blocks.append(
            f"[{index}] 来源: {source_name}\n"
            f"内容: {candidate.text[:800]}"
        )

    candidate_text = "\n\n".join(blocks)
    return (
        "你是检索结果相关性评估器。请判断每个候选片段是否能直接帮助回答用户问题。\n"
        "只输出 JSON，不要输出 Markdown 或解释。\n"
        '输出格式：{"relevant_indices":[1,2]}\n'
        "使用 1-based 候选编号；只保留确实相关的候选。\n"
        "没有相关候选时输出空数组。\n\n"
        f"用户问题：\n{question}\n\n"
        f"候选片段：\n{candidate_text}"
    )

def build_rewrite_prompt(question: str, current_query: str, candidates: list) -> str:
    """构造检索失败后的查询改写 Prompt。"""
    snippets = []
    for index, candidate in enumerate(candidates[:5], start=1):
        snippets.append(f"[{index}] {candidate.text[:200]}")

    candidate_text = "\n".join(snippets) or "无候选结果"
    return (
        "你是检索查询改写器。当前查询没有找到足够相关的资料。\n"
        "请保留用户原始意图，补充同义表达、关键术语或上下文，生成一个更适合检索的新查询。\n"
        "不要回答问题，不要解释，只输出 JSON。\n"
        '输出格式：{"query":"新的检索查询"}\n\n'
        f"原始问题：\n{question}\n\n"
        f"当前查询：\n{current_query}\n\n"
        f"上次候选片段：\n{candidate_text}"
    )

def build_direct_answer_prompt() -> str:
    """不需要检索时的对话 system prompt。"""
    return (
        "你是文档问答助手的对话层。用户当前的问题不需要检索知识库。\n"
        "请简洁、自然地回答，不要伪造文档引用，也不要声称答案来自知识库。"
    )
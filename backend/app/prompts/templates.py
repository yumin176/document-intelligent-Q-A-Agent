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
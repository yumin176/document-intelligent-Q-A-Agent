"""Rewrite：检索失败后生成新的检索查询。"""

import json
import logging
import re

from app.llm.base import ChatModel
from app.prompts.templates import build_rewrite_prompt
from app.vectorstore.base import SearchResult

logger = logging.getLogger(__name__)


def parse_rewrite_query(raw: str) -> str:
    """解析模型输出的新查询。

    要求：
    - 支持 JSON；
    - 支持 Markdown 代码块中的纯文本；
    - 不接受裸文本；
    - 无法解析时抛出 ValueError。
    """
    try:
        text = raw.strip()
        start_index = raw.find("{")
        end_index = raw.rfind("}")
        # 情况 1：存在 JSON 对象
        if start_index != -1 and end_index > start_index:
            # 处理json对象
            query_msg = json.loads(raw[start_index : end_index + 1])
            if not isinstance(query_msg, dict):
                raise ValueError("Rewrite 输出必须是对象")

            new_query = query_msg.get("query")
            if not isinstance(new_query, str):
                raise ValueError("query 必须是字符串")
            if not new_query.strip():
                raise ValueError("query不能为空")
            return new_query.strip()
        # 情况 2：Markdown 代码块
        elif text.startswith("```"):
            query = text
            query = re.sub(r"^```(?:json|text)?\s*", "", query)
            query = re.sub(r"\s*```$", "", query).strip()
            query = query.strip().strip('"').strip("'").strip()
            if not query:
                raise ValueError("query不能为空")
            return query
        # 情况 3：其他裸文本
        else:
            raise ValueError("Rewrite 输出不是 JSON 或 Markdown 文本")
    except (ValueError, TypeError) as exc:
        logger.warning("Rewrite 输出无效: %s", exc)
        raise ValueError("无法解析 query") from exc


def rewrite_query(
    question: str,
    current_query: str,
    candidates: list[SearchResult],
    chat_model: ChatModel,
) -> str:
    """生成新的检索查询，异常时使用保守回退。

    回退规则：
    - 如果 current_query 与原始问题不同，回退到原始问题；
    - 如果 current_query 已经是原始问题，回退为 "{question} 相关概念"；
    - 保证返回查询非空，且不要返回与 current_query 完全相同的查询。
    """
    prompt=build_rewrite_prompt(question,current_query,candidates)
    try:
        response=chat_model.complete(prompt=prompt, temperature=0)
        new_query=parse_rewrite_query(response)
        if new_query==current_query:
            return question if current_query!=question else f"{question} 相关概念"
        return new_query
    except Exception as exc:
        # 回退
        logger.warning("rewrite重写问题失败: %s", exc)
        return question if current_query!=question else f"{question} 相关概念"

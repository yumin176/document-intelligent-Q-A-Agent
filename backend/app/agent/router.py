"""Router 节点：判断走 direct 还是 retrieve。"""

import json
import re

from app.agent.state import Route
from app.llm.base import ChatModel
from app.prompts.templates import build_router_prompt


DIRECT_MARKERS = {
    "你好",
    "您好",
    "hi",
    "hello",
    "谢谢",
    "多谢",
    "你是谁",
    "你能做什么",
    "介绍一下你自己",
}

import logging

logger = logging.getLogger(__name__)

def parse_route(raw: str) -> Route:
    """从模型输出中解析 route。

    要求：
    - 兼容纯 JSON、Markdown 代码块和 JSON 前后的说明文字；
    - route 只能是 retrieve 或 direct；
    - 无法解析时抛出 ValueError。
    """
    try:
        start_index=raw.find("{")
        end_index=raw.rfind("}")
        route_msg=json.loads(raw[start_index:end_index+1])
        if route_msg["route"] not in ["direct","retrieve"]:
            raise ValueError("无法解析")
        return route_msg["route"]
    except (ValueError, KeyError, TypeError) as exc:
        logger.warning("Router 输出无效: %s", exc)
        # from exc 把原异常挂在新异常的__cause__ 上，保留异常链
        raise ValueError("无法解析 route") from exc


def heuristic_route(question: str) -> Route:
    """模型调用或解析失败时的保守回退。

    规则：
    - 明显寒暄走 direct；
    - 其他所有问题走 retrieve。
    """
    question=question.strip().lower()
    parts=[x for x in re.split(r"[\s，。！？、,.!?]+", question) if x]
    return "direct" if all(part in DIRECT_MARKERS for part in parts) else "retrieve"


def route_question(question: str, chat_model: ChatModel) -> Route:
    """调用 LLM 判断路由，失败时使用 heuristic_route。"""
    try:
        response=chat_model.complete(
            build_router_prompt(question),
            temperature=0
        )
        return parse_route(response)
    except Exception as err:
        return heuristic_route(question=question)

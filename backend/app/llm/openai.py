"""OpenAI 兼容的 Chat 模型实现。"""

from openai import OpenAI

from app.config import get_settings
from app.llm.base import ChatMessage, ChatModel


class OpenAIChatModel(ChatModel):
    """调用 OpenAI 兼容的 chat.completions 接口。

    默认复用 Embedding 的 base_url 和 api_key，
    仅需在 .env 中配置 LLM_MODEL。
    """

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        timeout: float | None = None,
    ) -> None:
        settings = get_settings()
        self.base_url = (
            base_url or settings.llm_base_url or settings.embedding_base_url
        ).rstrip("/")
        self.api_key = (
            api_key or settings.llm_api_key or settings.embedding_api_key
        )
        self.model = model or settings.llm_model
        self.default_temperature = settings.llm_temperature
        self.timeout = timeout if timeout is not None else settings.llm_timeout

        if not self.api_key:
            raise RuntimeError(
                "LLM_API_KEY 未配置；若无单独 LLM_API_KEY，将复用 EMBEDDING_API_KEY"
            )

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
        )

    def chat(
        self,
        messages: list[ChatMessage],
        temperature: float | None = None,
    ) -> str:
        payload = [{"role": message.role, "content": message.content} for message in messages]
        response = self.client.chat.completions.create(
            model=self.model,
            messages=payload,
            temperature=self.default_temperature if temperature is None else temperature,
        )
        content = response.choices[0].message.content
        return content or ""
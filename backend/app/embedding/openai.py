"""OpenAI 兼容的 Embedding 实现。"""


from app.config import get_settings
from app.embedding.base import Embedder
from openai import OpenAI


class OpenAIEmbedder(Embedder):
    """调用 OpenAI 兼容的 /embeddings 接口。

    支持 OpenAI、DeepSeek、通义千问、硅基流动等只要兼容
    OpenAI API 格式的服务，只需在 .env 中修改 base_url 和 model。
    """

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        timeout: float | None = None,
    ) -> None:
        settings = get_settings()
        self.base_url = (base_url or settings.embedding_base_url).rstrip("/")
        self.api_key = api_key or settings.embedding_api_key
        self.model = model or settings.embedding_model
        self.batch_size = settings.embedding_batch_size
        self.timeout = timeout if timeout is not None else settings.embedding_timeout
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )

    def embed(self, text: str) -> list[float]:
        self._ensure_api_key()
        completion = self.client.embeddings.create(
            model=self.model,
            input=text,
            timeout=self.timeout
        )
        vector=completion.data[0].embedding
        return vector

    def embed_batch(self, texts: list[str],batch_size:int| None =None) -> list[list[float]]:
        # 返回milvus存储所需格式
        if not texts:
            return []
        self._ensure_api_key()
        embeddings: list[list[float]] = []
        batch_size = batch_size or self.batch_size
        if batch_size <= 0:
            raise ValueError("batch_size 必须大于 0")
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            response = self.client.embeddings.create(
                model=self.model,
                input=batch,
                timeout=self.timeout,
            )
            items = sorted(response.data, key=lambda item: item.index)
            embeddings.extend(item.embedding for item in items)
        if len(embeddings) != len(texts):
            raise RuntimeError(
                f"Embedding 数量不匹配: 输入 {len(texts)} 条，返回 {len(embeddings)} 条"
            )
        return embeddings

    def _ensure_api_key(self) -> None:
        if not self.api_key:
            raise RuntimeError("EMBEDDING_API_KEY 未配置，请在 .env 中填写")

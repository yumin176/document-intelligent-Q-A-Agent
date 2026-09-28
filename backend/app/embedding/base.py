"""Embedding 抽象接口。"""

from abc import ABC, abstractmethod


class Embedder(ABC):
    """文本向量化接口。

    上层检索模块只依赖这个接口，不依赖具体实现。
    这样后续可以在 API / 本地模型之间自由切换。
    """

    @abstractmethod
    def embed(self, text: str) -> list[float]:
        """把单条文本转换为向量。"""

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """默认的批量实现：逐条调用 embed。子类可以覆盖以提高效率。"""
        return [self.embed(text) for text in texts]
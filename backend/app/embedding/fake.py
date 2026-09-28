"""测试用的确定性向量化实现。"""

import math

from app.embedding.base import Embedder


class FakeEmbedder(Embedder):
    """基于字符哈希的确定性向量，仅用于单元测试，不具备语义能力。

    它保证：
    - 相同文本永远得到相同向量；
    - 输出向量长度固定；
    - 向量已做 L2 归一化。
    """

    def __init__(self, dim: int = 32) -> None:
        self.dim = dim

    def embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dim
        for char in text:
            vector[ord(char) % self.dim] += 1.0
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]
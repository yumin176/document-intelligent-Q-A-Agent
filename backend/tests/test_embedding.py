"""Embedding 接口与 FakeEmbedder 的单元测试。"""

import math

from app.embedding.fake import FakeEmbedder


def test_embed_returns_fixed_dimension() -> None:
    embedder = FakeEmbedder(dim=8)
    vector = embedder.embed("hello")
    assert len(vector) == 8


def test_embed_is_deterministic() -> None:
    embedder = FakeEmbedder(dim=8)
    assert embedder.embed("hello") == embedder.embed("hello")


def test_embed_is_normalized() -> None:
    embedder = FakeEmbedder(dim=8)
    vector = embedder.embed("hello")
    norm = math.sqrt(sum(value * value for value in vector))
    assert math.isclose(norm, 1.0, rel_tol=1e-6)


def test_embed_batch_returns_same_count() -> None:
    embedder = FakeEmbedder(dim=8)
    result = embedder.embed_batch(["a", "b", "c"])
    assert len(result) == 3
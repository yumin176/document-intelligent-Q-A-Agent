import logging

import jieba
from rank_bm25 import BM25Okapi
import numpy as np

from app.retrieval.base import Retriever
from app.vectorstore.base import SearchResult, TextRecord, VectorStore

jieba.setLogLevel(logging.WARNING)

def tokenize(text: str) -> list[str]:
    tokens = []
    for raw_token in jieba.lcut(text):
        token = raw_token.strip().lower()
        if not token:
            continue
        #isalnum() 用于判断字符串是否只由字母和数字组成，如果是则返回 True，否则返回 False。
        # 这一步是为了过滤 , . 。之类的标点，防止噪声过大
        if token.isalnum() or "_" in token:
            tokens.append(token)
    return tokens

# bm25检索
class BM25Retriever(Retriever):
    def __init__(self,vector_store: VectorStore):
        self.vector_store = vector_store
        self.all_chunks: list[TextRecord] = []
        self.bm25: BM25Okapi | None = None
        self.refresh()
    # BM25 需要维护文档索引，文档更新后必须刷新
    def refresh(self) -> None:
        """重新构建 BM25 索引，文档更新后调用。"""
        self.all_chunks = self.vector_store.all_chunks()
        tokenized_corpus = [tokenize(chunk.text) for chunk in self.all_chunks]

        if not tokenized_corpus:
            self.bm25 = None
            return
        self.bm25 = BM25Okapi(tokenized_corpus)


    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        # 查询分词
        tokenized_query = tokenize(query)
        if top_k <= 0 or self.bm25 is None or not self.all_chunks:
            return []
        # 获取分数
        scores=self.bm25.get_scores(tokenized_query)
        # argsort函数返回数组元素从小到大排序后的索引，而不是排序后的数组本身
        # [::-1] 把序列反转
        indices = np.argsort(scores)[::-1]
        results: list[SearchResult] = []

        for index in indices:
            score = float(scores[index])
            # 过滤掉得分为0（没有匹配上）的结果
            if score <= 0:
                continue

            chunk = self.all_chunks[index]
            results.append(
                SearchResult(
                    id=chunk.id,
                    score=score,
                    text=chunk.text,
                    metadata=chunk.metadata,
                )
            )

            if len(results) >= top_k:
                break

        return results

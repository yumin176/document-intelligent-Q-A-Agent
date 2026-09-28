from app.vectorstore.base import ChunkRecord, SearchResult, VectorStore
import math

def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)

class InMemoryVectorStore(VectorStore):
    def __init__(self) -> None:
        self._records: dict[str, ChunkRecord] = {}

    def upsert(self, records: list[ChunkRecord]) -> None:
        # TODO: 把每条记录存进 self._records，以 id 为 key
        for record in records:
            self._records[record.id]=record


    def search(
        self,
        vector: list[float],
        top_k: int = 5,
        filter_expr: str | None = None,
    ) -> list[SearchResult]:
        # TODO: 计算查询向量与每条记录 embedding 的余弦相似度
        # 按分数从高到低排序，返回前 top_k 个
        # 校验如果top_k小于等于0，库存为空，返回空数组
        if top_k<=0 or (not self._records):
            return []
        score_list=[]
        for id,record in self._records.items():
            score=_cosine(vector,record.embedding)
            score_list.append({
                "score":score,
                "record":record
            })
        target_chunk_list=sorted(score_list,key=lambda x:x["score"], reverse=True)[:top_k]
        result=[
            SearchResult(
                id=chunk["record"].id,
                score=chunk["score"],
                text=chunk["record"].text,
                metadata=chunk["record"].metadata
            ) for chunk in target_chunk_list
        ]
        return result


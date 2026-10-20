from app.retrieval.bm25 import BM25Retriever
from app.vectorstore.base import ChunkRecord
from app.vectorstore.memory import InMemoryVectorStore
def test_bm25_retriever():
    store=InMemoryVectorStore()
    store.upsert([
            ChunkRecord(id="a", text="var声明的变量可以重复声明", embedding=[1.0, 0.0, 0.0]),
            ChunkRecord(id="b", text="const和let声明的变量不会变量提升，在声明之前访问变量会报错，暂时性死区", embedding=[0.0, 1.0, 0.0]),
            ChunkRecord(id="c", text="const 声明的变量声明时就得赋值", embedding=[0.0, 0.0, 1.0]),
        ])
    bm25=BM25Retriever(
        vector_store=store
    )
    query="const和let"
    res=bm25.search(query)
    print(res)
    assert len(res) >= 1
    assert res[0].id == "b"   # b 同时包含 const 和 let
    assert all(result.score > 0 for result in res)

def test_empty_corpus_returns_empty() -> None:
    retriever = BM25Retriever(InMemoryVectorStore())
    assert retriever.search("const") == []


def test_no_match_returns_empty() -> None:
    store=InMemoryVectorStore()
    store.upsert([
            ChunkRecord(id="a", text="var声明的变量可以重复声明", embedding=[1.0, 0.0, 0.0]),
        ])
    bm25=BM25Retriever(
        vector_store=store
    )
    query="const和let"
    res=bm25.search(query)
    assert res==[]



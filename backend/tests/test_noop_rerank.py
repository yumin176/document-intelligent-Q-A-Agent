from app.rerank.noop import NoOpReranker
from app.vectorstore.base import SearchResult

def test_noop_rerank():
    reranker=NoOpReranker()
    candidates=[
        SearchResult(id="a",score=0.85,text="var声明的变量有变量提升"),
        SearchResult(id="b",score=0.75,text="var声明的变量可以重复声明"),
        SearchResult(id="c",score=0.51,text="const和let声明的变量不会变量提升")
    ]
    result=reranker.rerank(query="var",candidates=candidates,top_k=2)
    assert len(result) == 2
    assert [item.id for item in result] == ["a","b"]

def test_noop_rerank_top_k():
    reranker=NoOpReranker()
    candidates=[
        SearchResult(id="a",score=0.85,text="var声明的变量有变量提升"),
        SearchResult(id="b",score=0.75,text="var声明的变量可以重复声明"),
        SearchResult(id="c",score=0.51,text="const和let声明的变量不会变量提升")
    ]
    assert len(reranker.rerank(query="var",candidates=candidates,top_k=-2)) == 0

def test_noop_rerank_empty_candidates():
    reranker=NoOpReranker()
    assert len(reranker.rerank(query="var",candidates=[],top_k=-2)) == 0

from app.embedding.base import Embedder
from app.retrieval.base import Retriever
from app.vectorstore.base import SearchResult, VectorStore

# 向量检索
class VectorRetriever(Retriever):
    def __init__(self, embedder: Embedder, vector_store: VectorStore) -> None:
        self.embedder=embedder
        self.vector_store=vector_store
    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        vector = self.embedder.embed(query)
        return self.vector_store.search(vector, top_k=top_k)

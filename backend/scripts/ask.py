"""命令行 RAG 问答演示。

用法：
    python scripts/ask.py "const和let有什么区别"
    python scripts/ask.py --top-k 5 "公司年假有多少天"
"""

import argparse
import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.embedding.openai import OpenAIEmbedder
from app.llm.openai import OpenAIChatModel
from app.rag.service import RAGService
from app.vectorstore.milvus import MilvusStore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RAG 问答")
    parser.add_argument("question", help="要提问的问题")
    parser.add_argument("--top-k", type=int, default=4, help="检索 top-k 个 chunk")
    return parser.parse_args()


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    args = parse_args()

    embedder = OpenAIEmbedder()
    chat_model = OpenAIChatModel()
    store = MilvusStore()

    try:
        service = RAGService(
            embedder=embedder,
            vector_store=store,
            chat_model=chat_model,
            top_k=args.top_k,
        )
        result = service.answer(args.question, top_k=args.top_k)

        print("=" * 70)
        print(f"问题：{args.question}")
        print("=" * 70)
        print("回答：")
        print(result.answer)

        if result.sources:
            print("\n引用来源：")
            for index, source in enumerate(result.sources, start=1):
                filename = source.metadata.get("filename") or source.chunk_id
                chunk_index = source.metadata.get("chunk_index")
                print(f"[{index}] {filename} | chunk {chunk_index} | score={source.score:.4f}")
                print(f"    {source.text[:120]}")
        else:
            print("\n未检索到任何资料。")
    finally:
        store.close()


if __name__ == "__main__":
    main()
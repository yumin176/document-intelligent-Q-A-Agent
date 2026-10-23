"""问答接口"""
from fastapi import APIRouter,Depends
from app.schemas import ChatResponse,ChatRequest,SourceOut
from app.dependencies import get_agent_service
from app.agent.service import AgentService
import json
from fastapi.responses import StreamingResponse

router = APIRouter(prefix="/api",tags=["chat"])

def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"

@router.post("/chat", response_model=ChatResponse)
def chat(body:ChatRequest,service:AgentService=Depends(get_agent_service))->ChatResponse:
    result=service.answer(
        question=body.question,
        top_k=body.top_k
    )
    return ChatResponse(
        answer=result.answer,
        sources=[
            SourceOut(
                chunk_id=source.chunk_id,
                score=source.score,
                text=source.text,
                metadata=source.metadata
            ) for source in result.sources
        ]
    )

@router.post("/chat/stream")
def chat_stream(
    body: ChatRequest,
    service: AgentService = Depends(get_agent_service),
):
    def generate():
        try:
            token_iter, sources = service.answer_stream(body.question, body.top_k)

            # 先发引用来源
            # model_dump 将模型实例转换为标准python字典
            source_payload = [
                SourceOut(
                    chunk_id=s.chunk_id,
                    score=s.score,
                    text=s.text,
                    metadata=s.metadata,
                ).model_dump()
                for s in sources
            ]
            yield _sse("sources", {"sources": source_payload})

            # 再流式发 token
            for token in token_iter:
                yield _sse("delta", {"content": token})

            # 最后发完成事件
            yield _sse("done", {})

        except Exception as exc:
            yield _sse("error", {"message": str(exc)})

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache"},
    )

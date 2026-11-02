"""问答接口"""
import json
import uuid

from fastapi import APIRouter,Depends
from app.schemas import ChatResponse,ChatRequest,SourceOut
from app.dependencies import get_agent_service
from app.agent.service import AgentService
from app.observability.tracing import record_output, trace_request
from fastapi.responses import StreamingResponse
from starlette.concurrency import run_in_threadpool

router = APIRouter(prefix="/api",tags=["chat"])

def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"

@router.post("/chat", response_model=ChatResponse)
def chat(body:ChatRequest,service:AgentService=Depends(get_agent_service))->ChatResponse:
    # 根 span：整个请求一条 trace；未配置 LangSmith 时透传
    with trace_request(body.question, body.top_k, uuid.uuid4().hex[:12]) as root:
        result=service.answer(
            question=body.question,
            top_k=body.top_k
        )
        record_output(
            root,
            sources=len(result.sources),
            answer_chars=len(result.answer),
        )
    return ChatResponse(
        answer=result.answer,
        sources=[SourceOut.from_source(source) for source in result.sources],
    )

@router.post("/chat/stream")
async def chat_stream(
    body: ChatRequest,
    service: AgentService = Depends(get_agent_service),
):
    async def generate():
        request_id = uuid.uuid4().hex[:12]
        try:
            # 根 span 包住整个流式过程。端点写成 async 让生成器始终跑在
            # 事件循环的同一个上下文里：显式开线程池反而会导致 with 的
            # 进入/退出跨线程，trace 根 span 上报不出去。
            with trace_request(body.question, body.top_k, request_id) as root:
                # 检索 + 打分是阻塞调用（含远端 HTTP），丢线程池避免卡住事件循环；
                # anyio 会复制当前上下文进线程，节点 span 因此能挂到根 span 下
                token_iter, sources = await run_in_threadpool(
                    service.answer_stream, body.question, body.top_k
                )

                # 先发引用来源
                # model_dump 将模型实例转换为标准python字典
                source_payload = [
                    SourceOut.from_source(s).model_dump() for s in sources
                ]
                yield _sse("sources", {"sources": source_payload})

                # 再流式发 token（每次一个网络读，阻塞很轻）
                chars = 0
                for token in token_iter:
                    chars += len(token)
                    yield _sse("delta", {"content": token})

                record_output(
                    root,
                    sources=len(source_payload),
                    answer_chars=chars,
                )
                # 最后发完成事件
                yield _sse("done", {})

        except Exception as exc:
            yield _sse("error", {"message": str(exc)})

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache"},
    )

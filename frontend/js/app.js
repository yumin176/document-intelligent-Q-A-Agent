// 获取dom
// 聊天发送按钮
const chat_btn=document.getElementById('bottom_box_btn')
// 聊天框
const chat_input=document.getElementById("input_box")
// 聊天滚动容器
const chat_scroll=document.querySelector(".top_chat_box")
// 聊天内容区（一轮轮问答都插到这里，便于居中定宽）
const chat_box=document.querySelector(".chat_inner")
// 欢迎区
const welcome=document.getElementById("welcome")
//文件选择器
const document_select=document.getElementById("document")
// 文件上传按钮
const document_upload_btn=document.getElementById("document_btn")
// 文件选择器的文字
const file_picker=document.getElementById("file_picker")
const file_picker_text=document.getElementById("file_picker_text")
// 文件列表容器
const documents_wrap=document.querySelector(".documents_wrap")
// 文档数量
const doc_count=document.getElementById("doc_count")
// 本次提问历史
const history_section=document.getElementById("history_section")
const history_count=document.getElementById("history_count")
const history_list=document.getElementById("history_list")

// 会话状态：只存当前标签页，刷新或关闭页面即失效
const STORAGE_KEY="smartqa.chat.v1"

// 对话轮次：{id, question, answer, sources, status}
let turns=[]
let turnSeq=0
let cards=null
let streaming=false

// ---------- 会话状态持久化 ----------

// 从知识库页返回是 navigate / back_forward，刷新页面是 reload。
// 只有前者恢复对话，刷新按需求清空。
function shouldRestoreSession(){
    const entries=performance.getEntriesByType?performance.getEntriesByType("navigation"):null
    const type=entries&&entries.length?entries[0].type:"navigate"
    return type!=="reload"
}

function saveSession(){
    try {
        sessionStorage.setItem(STORAGE_KEY,JSON.stringify({turns}))
    } catch (error) {
        console.log("会话保存失败：",error)
    }
}

function clearSession(){
    try {
        sessionStorage.removeItem(STORAGE_KEY)
    } catch (error) {
        console.log("会话清理失败：",error)
    }
}

function loadSession(){
    if (!shouldRestoreSession()){
        clearSession()
        return []
    }
    try {
        const raw=sessionStorage.getItem(STORAGE_KEY)
        if (!raw) return []
        const data=JSON.parse(raw)
        return Array.isArray(data.turns)?data.turns:[]
    } catch (error) {
        console.log("会话恢复失败：",error)
        return []
    }
}

// ---------- 工具函数 ----------

// 跳转到知识库页面：doc 定位文档，anchor 定位到具体 chunk
function openKnowledge({doc='',anchor=''}={}){
    const params=new URLSearchParams()
    if (doc) params.set("doc",doc)
    if (anchor) params.set("anchor",anchor)
    const query=params.toString()
    window.location.href="/knowledge#/knowledge"+(query?"?"+query:"")
}

function truncate(text="",limit=90){
    const flat=text.replace(/\s+/g," ").trim()
    return flat.length>limit?flat.slice(0,limit)+"…":flat
}

// 滚动到底部：用户手动上翻查看历史时不打断
function scrollToBottom(force=false){
    if (!chat_scroll) return
    const nearBottom=chat_scroll.scrollHeight-chat_scroll.scrollTop-chat_scroll.clientHeight<140
    if (force||nearBottom){
        chat_scroll.scrollTop=chat_scroll.scrollHeight
    }
}

// 输入框随内容自动增高
function autoGrowInput(){
    chat_input.style.height="auto"
    chat_input.style.height=Math.min(chat_input.scrollHeight,160)+"px"
}

// 打字指示器：等第一个 token 到达后移除
function showTyping(container){
    const typing=document.createElement("div")
    typing.className="typing"
    typing.append(document.createElement("span"),document.createElement("span"),document.createElement("span"))
    container.append(typing)
    return typing
}

// 打字机：把流式到达的文本按节奏吐到页面上。
// 积压越多吐得越快；流结束后进入加速模式，排空了再回调（用于插入来源卡片）。
function createTypewriter(element,onDrain){
    let pending=""
    let shown=""
    let timer=null
    let finished=false
    let drained=false
    const TICK=20        // 每帧 20ms
    const BASE=1.4       // 基础速度：约 70 字/秒，短回答能看清逐字出现
    const CAP=14         // 单帧上限，避免长回答收尾时糊成一团

    function drain(){
        if (drained) return
        drained=true
        if (typeof onDrain==="function") onDrain()
    }

    function tick(){
        if (!pending){
            timer=null
            if (finished) drain()
            return
        }
        // 积压变多才提速（追上网络），否则保持稳定的打字节奏
        let chars=BASE
        if (pending.length>120){
            chars=Math.min(CAP,Math.round(BASE+pending.length/70))
        }
        if (finished) chars=Math.min(CAP,chars*1.6)
        chars=Math.max(1,Math.round(chars))
        shown+=pending.slice(0,chars)
        pending=pending.slice(chars)
        element.textContent=shown
        scrollToBottom()
        timer=setTimeout(tick,TICK)
    }

    return {
        push(text){
            if (!text) return
            pending+=text
            if (timer===null) timer=setTimeout(tick,TICK)
        },
        // 流结束：加速把剩余文字吐完，排空后回调
        finish(){
            if (finished) return
            finished=true
            if (timer===null){
                if (pending) timer=setTimeout(tick,TICK)
                else drain()
            }
            // 兜底：极端情况下 4s 内没排空就一次性补齐
            setTimeout(()=>{
                if (!drained){
                    shown+=pending
                    pending=""
                    element.textContent=shown
                    drain()
                }
            },4000)
        },
        // 立即显示全文（不用于正常流程，供异常兜底）
        flush(){
            if (timer!==null){
                clearTimeout(timer)
                timer=null
            }
            shown+=pending
            pending=""
            element.textContent=shown
            drain()
            scrollToBottom()
        },
        get text(){
            return shown
        },
    }
}

// ---------- 渲染 ----------

// 渲染用户气泡（返回元素，由调用方插入 turn 容器）
function renderUserChat(content){
    const div=document.createElement("div")
    div.classList.add("user_chat_frame","chat_frame")
    div.textContent=content
    return div
}

// 渲染来源卡片：点击跳转到知识库并高亮对应锚点
function renderSources(sources=[]){
    const card_container=document.createElement("div")
    card_container.classList.add("source_card_container")
    sources.forEach((source,index)=>{
        const card=document.createElement("button")
        card.type="button"
        card.className="source_card source_card_action"

        const title=document.createElement("div")
        title.className='source_title'
        const tab=document.createElement("span")
        tab.className="Index_tab"
        tab.textContent=`[${index+1}]`
        const name=document.createElement("span")
        name.textContent=source.filename||source?.metadata?.filename||'未知来源'
        title.append(tab,name)

        const metadata=document.createElement("div")
        metadata.className='source_metadata'
        const chunkIndex=source.chunk_index??source?.metadata?.chunk_index??0
        const bits=[`第 ${Number(chunkIndex)+1} 块`]
        if (source.page_number) bits.push(`第 ${source.page_number} 页`)
        bits.push(`相关度 ${Number(source.score||0).toFixed(3)}`)
        metadata.textContent=bits.join(" · ")

        const preview=document.createElement("div")
        preview.className="source_preview"
        preview.textContent=truncate(source.text||"")

        const hint=document.createElement("div")
        hint.className="source_hint"
        hint.textContent="点击查看原文"

        card.append(title,metadata,preview,hint)
        card.onclick=()=>openKnowledge({
            doc:source.document_id||"",
            anchor:source.anchor_id||source.chunk_id||"",
        })
        card_container.append(card)
    })
    return card_container
}

// 渲染一整轮问答（历史恢复与实时对话共用）
function renderTurn(turn){
    const turnEl=document.createElement("div")
    turnEl.className="turn"
    turnEl.dataset.turnId=String(turn.id)

    turnEl.append(renderUserChat(turn.question))

    const assistant=document.createElement("div")
    assistant.classList.add("assistant_chat_frame","chat_frame")
    const contentEl=document.createElement("div")
    if (turn.status==="interrupted"){
        contentEl.textContent="（这轮回答在离开页面时被中断）"
        contentEl.style.color="var(--text-muted)"
    } else {
        contentEl.textContent=turn.answer||""
    }
    if (turn.status==="error"){
        contentEl.style.color="var(--text-muted)"
    }
    assistant.append(contentEl)
    if (turn.sources&&turn.sources.length){
        assistant.append(renderSources(turn.sources))
    }
    turnEl.append(assistant)

    chat_box.append(turnEl)
    return {turnEl,assistant,contentEl}
}

// 侧栏本次提问列表
function renderHistory(){
    history_list.innerHTML=""
    if (!turns.length){
        history_section.hidden=true
        history_list.hidden=true
        history_count.textContent=""
        return
    }
    history_section.hidden=false
    history_list.hidden=false
    history_count.textContent=`${turns.length} 条`
    turns.forEach((turn,index)=>{
        const item=document.createElement("button")
        item.type="button"
        item.className="history_item"

        const num=document.createElement("span")
        num.className="history_index"
        num.textContent=String(index+1)

        const text=document.createElement("span")
        text.className="history_text"
        text.textContent=turn.question

        item.append(num,text)
        item.title=turn.question
        item.onclick=()=>focusTurn(turn.id)
        history_list.append(item)
    })
}

// 点击历史：滚动到对应轮次并短暂高亮
function focusTurn(turnId){
    const target=chat_box.querySelector(`.turn[data-turn-id="${turnId}"]`)
    if (!target) return
    target.scrollIntoView({behavior:"smooth",block:"start"})
    target.classList.add("turn--active")
    setTimeout(()=>target.classList.remove("turn--active"),1400)
}

// ---------- 事件 ----------

// 绑定事件
function bindEvent(){
    chat_btn.onclick=handelSendChat
    document_upload_btn.onclick=handelUploadFile

    // 选择文件后显示文件名
    document_select.onchange=()=>{
        const file=document_select.files[0]
        if (!file){
            file_picker_text.textContent="选择文件"
            file_picker.classList.remove("file_picker--filled")
            return
        }
        file_picker_text.textContent=file.name
        file_picker.classList.add("file_picker--filled")
    }

    // Enter 发送，Shift + Enter 换行；输入法组字期间不拦截
    chat_input.addEventListener("keydown",(event)=>{
        if (event.key==="Enter"&&!event.shiftKey&&!event.isComposing){
            event.preventDefault()
            handelSendChat()
        }
    })
    chat_input.addEventListener("input",autoGrowInput)

    // 示例问题：点一下直接发送
    document.querySelectorAll(".example_btn").forEach((btn)=>{
        btn.addEventListener("click",()=>{
            chat_input.value=btn.dataset.question||""
            autoGrowInput()
            handelSendChat()
        })
    })
}

// 渲染上传状态条目
function renderDocument(title,state){
    const document_item=document.createElement("div")
    document_item.className="document"
    const title_div=document.createElement("div")
    title_div.className="document_name"
    title_div.textContent=title
    const state_div=document.createElement("div")
    state_div.className="document_meta"
    state_div.textContent=state
    document_item.append(title_div,state_div)
    documents_wrap.prepend(document_item)
    return state_div
}

// 渲染知识库文档列表（点击进入只读查看页）
async function loadDocuments(){
    try {
        const res=await fetch('/api/knowledge/documents')
        if (!res.ok) throw new Error(`HTTP ${res.status}`)
        const docs=await res.json()
        documents_wrap.innerHTML=""
        doc_count.textContent=docs.length?`${docs.length} 篇`:""
        if (!docs.length){
            const empty=document.createElement("div")
            empty.className="documents_empty"
            empty.textContent="还没有文档，上传后即可在这里查看分块"
            documents_wrap.append(empty)
            return
        }
        for (const doc of docs){
            const item=document.createElement("button")
            item.type="button"
            item.className="document document_action"
            const name=document.createElement("div")
            name.className="document_name"
            name.textContent=doc.doc_name||doc.filename||doc.document_id
            const meta=document.createElement("div")
            meta.className="document_meta"
            meta.textContent=`${doc.chunk_count} 块`
            item.append(name,meta)
            item.onclick=()=>openKnowledge({doc:doc.document_id})
            documents_wrap.append(item)
        }
    } catch (error) {
        documents_wrap.innerHTML=""
        const failed=document.createElement("div")
        failed.className="documents_empty"
        failed.textContent="文档列表加载失败，请确认服务已启动"
        documents_wrap.append(failed)
        console.log("文档列表加载失败：",error)
    }
}

// 发送聊天
async function handelSendChat(){
    const question=chat_input.value.trim()
    if (!question) {
        chat_input.focus()
        return
    }
    if (streaming) return

    // 首次提问时收起欢迎区
    if (welcome&&welcome.parentNode) welcome.remove()

    chat_input.value=''
    autoGrowInput()

    const turn={id:++turnSeq,question,answer:"",sources:[],status:"streaming"}
    turns.push(turn)
    saveSession()
    renderHistory()

    const {assistant,contentEl}=renderTurn(turn)
    const typingEl=showTyping(assistant)
    // 正文吐完（排空）后再插入来源卡片，避免卡片先于文字出现
    const typer=createTypewriter(contentEl,()=>{
        if (cards) assistant.append(cards)
        scrollToBottom()
    })
    scrollToBottom(true)

    try {
        streaming=true
        chat_btn.disabled=true
        chat_btn.textContent="生成中"
        cards=null
        await api_chat_stream_post(question,turn,typer,typingEl)
        // 单个 error 事件已把状态置为 error，这里不覆盖
        if (turn.status!=="error") turn.status="done"
        typer.finish()
        saveSession()
    } catch (error) {
        console.log(error)
        turn.status="error"
        if (!turn.answer) turn.answer="请求失败，请稍后重试"
        typer.flush()
        if (typingEl&&typingEl.parentNode) typingEl.remove()
        saveSession()
    } finally {
        streaming=false
        chat_btn.disabled=false
        chat_btn.textContent="发送"
        scrollToBottom()
    }
}

// 处理文件上传
async function handelUploadFile(){
    if(document_select.files.length<=0){
        alert("请先选择文件")
        return
    }
    const file=document_select.files[0]
    // 上传前先清掉空状态提示
    const empty=documents_wrap.querySelector(".documents_empty")
    if (empty) empty.remove()
    // 先创建文件dom
    const state_div=renderDocument(file.name,"正在上传…")
    document_upload_btn.disabled=true
    document_upload_btn.textContent="上传中…"
    // 上传文件
    try {
        await api_documents_post(file)
        state_div.textContent="上传成功，已入库"
        await loadDocuments()
        document_select.value=""
        file_picker_text.textContent="选择文件"
        file_picker.classList.remove("file_picker--filled")
    } catch (error) {
        state_div.textContent="上传失败"
    } finally {
        document_upload_btn.disabled=false
        document_upload_btn.textContent="上传并入库"
    }
}

// 文档上传接口
async function api_documents_post(file){
    const formData = new FormData();
    formData.append('file', file);
    const res=await fetch('/api/documents',{
      method: 'POST',
      body: formData
    })
    if (!res.ok) throw new Error(`HTTP ${res.status}`)
    const data = await res.json();
    console.log("文件上传：",data);
    return data
}

// 聊天接口
async function api_chat_stream_post(question,turn,typer,typingEl,top_k=4){
    const res = await fetch('/api/chat/stream', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question,top_k }),
    });

    if (!res.ok) throw new Error(`HTTP ${res.status}`);

    const reader = res.body.getReader();
    // 解码器
    const decoder = new TextDecoder('utf-8');
    let buffer = "";
    while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });

        // SSE 事件之间用空行分隔
        const parts = buffer.split("\n\n");
        buffer = parts.pop() || "";

        for (const part of parts) {
            handleSseEvent(part,turn,typer,typingEl);
        }
    }
    if (buffer.trim()) {
        handleSseEvent(buffer, turn, typer, typingEl)
    }
}

function handleSseEvent(raw,turn,typer,typingEl) {
  let event = "message";
  const dataLines = [];

  for (const line of raw.split("\n")) {
    if (line.startsWith("event:")) {
      event = line.slice(6).trim();
    } else if (line.startsWith("data:")) {
      dataLines.push(line.slice(5).trimStart());
    }
  }

  const data = JSON.parse(dataLines.join("\n") || "{}");
  if (event === "sources") {
    // 创建来源卡片dom（先挂在内存里，等正文吐完再插入）
    cards=renderSources(data.sources)
    turn.sources=data.sources||[]
  } else if (event === "delta") {
    if (typingEl&&typingEl.parentNode) typingEl.remove()
    typer.push(data.content||"")
    turn.answer+=data.content||""
  } else if (event === "done") {
    if (typingEl&&typingEl.parentNode) typingEl.remove()
  } else if (event === "error") {
    if (typingEl&&typingEl.parentNode) typingEl.remove()
    const message="抱歉，这次回答失败了："+(data.message||"未知错误")
    typer.push(message)
    turn.answer+=message
    turn.status="error"
  }
}

// 主函数
function main(){
    bindEvent()

    // 恢复会话：从知识库页返回时保留，刷新则清空
    turns=loadSession()
    if (turns.length){
        if (welcome&&welcome.parentNode) welcome.remove()
        turnSeq=turns.reduce((max,turn)=>Math.max(max,turn.id||0),0)
        // 离开页面时还没答完的轮次，标记为中断，避免恢复成空白气泡
        turns.forEach((turn)=>{
            if (turn.status==="streaming"){
                turn.status=turn.answer?"done":"interrupted"
            }
        })
        turns.forEach((turn)=>renderTurn(turn))
        renderHistory()
        // 等布局稳定后再贴到底部
        requestAnimationFrame(()=>scrollToBottom(true))
    }

    loadDocuments()
    chat_input.focus()
}

main()

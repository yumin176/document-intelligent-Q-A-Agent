// 获取dom
// 聊天发送按钮
const chat_btn=document.getElementById('bottom_box_btn')
// 聊天框
const chat_input=document.getElementById("input_box")
// 聊天窗口
const chat_box=document.querySelector(".top_chat_box")
//文件选择器
const document_select=document.getElementById("document")
// 文件上传按钮
const document_upload_btn=document.getElementById("document_btn")
// 文件列表容器
const documents_wrap=document.querySelector(".documents_wrap")

// 定义变量
let cards=null

// 绑定事件
function bindEvent(){
    chat_btn.onclick=handelSendChat
    document_upload_btn.onclick=handelUploadFile
}
// 渲染文件
function renderDocument(title,state){
    const document_item=document.createElement("div")
    document_item.className="document"
    const title_div=document.createElement("div")
    title_div.textContent=title
    const state_div=document.createElement("div")
    state_div.textContent=state
    document_item.append(title_div,state_div)
    documents_wrap.append(document_item)
    return state_div
}
// 渲染用户聊天框
function renderUserChat(content){
    const div=document.createElement("div")
    div.classList.add("user_chat_frame","chat_frame")
    div.textContent=content
    chat_box.append(div)
    chat_input.value=''
}
// 渲染来源卡片
function renderSources(sources=[]){
    // 来源卡片
    const card_container=document.createElement("div")
    card_container.classList.add("source_card_container")
    for (const source of sources){
        const card=document.createElement("div")
        card.className="source_card"
        const title=document.createElement("div")
        title.className='source_title'
        title.textContent=source?.metadata?.filename || '未知来源'
        const metadata=document.createElement("div")
        metadata.className='source_metadata'
        metadata.append(
            `chunk_index:${source.metadata.chunk_index}\n`,
            `score:${source.score}\n`,
            `text:${source.text}\n`
        )
        card.append(title,metadata)
        card_container.append(card)
    }
    return card_container
}
// 发送聊天
async function handelSendChat(){
    const input=chat_input.value
    if (!input){
        alert("消息不能为空")
        return
    }
    renderUserChat(input)
    // 请求接口
    try {
        // 先把来源卡片dom清空
        cards=null
        // 创建助手聊天框dom
        const assistant_chat=document.createElement("div")
        assistant_chat.classList.add("assistant_chat_frame","chat_frame")
        // 创建内容dom
        const contentEl = document.createElement("div")
        assistant_chat.append(contentEl)
        // 追加到聊天窗口
        chat_box.append(assistant_chat)
        await api_chat_stream_post(input,assistant_chat,contentEl)
    } catch (error) {
        console.log(error)
    }
}
// 处理文件上传
async function handelUploadFile(){
    if(document_select.files.length<=0){
        alert("请先选择文件")
        return
    }
    const file=document_select.files[0]
    // 先创建文件dom
    const state_div=renderDocument(file.name,"正在上传")
    // 上传文件
    try {
        const data=await api_documents_post(file)
        state_div.textContent="上传成功"
    } catch (error) {
        state_div.textContent="上传失败"
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
    const data = await res.json();
    console.log("文件上传：",data);
    return data
}
// 聊天接口
async function api_chat_stream_post(question,assistant_chat,contentEl,top_k=4){
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
            handleSseEvent(part,assistant_chat,contentEl);
        }
    }
    if (buffer.trim()) {
        handleSseEvent(buffer, assistant_chat, contentEl)
    }
}
function handleSseEvent(raw,assistant_chat,contentEl) {
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
    // 创建来源卡片dom
    cards=renderSources(data.sources)
    console.log(cards)
    console.log("*************sources************")
    console.log(data.sources)
  } else if (event === "delta") {
    contentEl.textContent+=data.content
    // console.log("*************content************")
    // console.log(data.content)
  } else if (event === "done") {
    // finishStreaming();
    // 追加到assistant_chat助手聊天框
    if (cards){
        assistant_chat.append(cards)
    }
    console.log("*************done************")
  } else if (event === "error") {
    // showError(data.message);
    alert(data.message)
  }
}

// 主函数
function main(){
    // console.log("main woking")
    bindEvent()
}

main()

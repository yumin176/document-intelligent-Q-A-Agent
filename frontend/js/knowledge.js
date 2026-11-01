const docListEl = document.getElementById("kb_doc_list");
const statsEl = document.getElementById("kb_stats");
const titleEl = document.getElementById("kb_main_title");
const metaEl = document.getElementById("kb_main_meta");
const chunkListEl = document.getElementById("kb_chunk_list");

const MAX_PREVIEW = 600;

function parseRoute() {
    const hash = window.location.hash || "";
    const markIndex = hash.indexOf("?");
    const query = new URLSearchParams(markIndex >= 0 ? hash.slice(markIndex + 1) : "");
    return {
        doc: query.get("doc") || "",
        anchor: query.get("anchor") || "",
    };
}

function buildHash(docId, anchorId) {
    const params = new URLSearchParams();
    if (docId) params.set("doc", docId);
    if (anchorId) params.set("anchor", anchorId);
    const query = params.toString();
    return "#/knowledge" + (query ? "?" + query : "");
}

async function requestJson(url) {
    const response = await fetch(url);
    if (!response.ok) {
        let detail = "HTTP " + response.status;
        try {
            const body = await response.json();
            if (body && body.detail) detail = body.detail;
        } catch (error) {
            detail = "HTTP " + response.status;
        }
        const wrapped = new Error(detail);
        wrapped.status = response.status;
        throw wrapped;
    }
    return response.json();
}

function formatDocMeta(doc) {
    const parts = [doc.chunk_count + " 块"];
    if (doc.chunk_strategy) {
        parts.push(doc.chunk_strategy + " " + doc.chunk_size + "/" + doc.chunk_overlap);
    }
    if (doc.embedding_model) {
        parts.push(doc.embedding_model + (doc.embedding_dimension ? " " + doc.embedding_dimension + "d" : ""));
    }
    return parts.join(" · ");
}

function renderDocList(docs, activeDocId) {
    docListEl.innerHTML = "";
    if (!docs.length) {
        const empty = document.createElement("div");
        empty.className = "kb_empty";
        empty.textContent = "知识库还没有文档。先在对话页上传一份 .txt / .md / .pdf，入库后会自动出现在这里。";
        docListEl.append(empty);
        return;
    }

    for (const doc of docs) {
        const item = document.createElement("button");
        item.type = "button";
        item.className = "kb_doc_item" + (doc.document_id === activeDocId ? " kb_doc_item--active" : "");

        const name = document.createElement("span");
        name.className = "kb_doc_name";
        name.textContent = doc.doc_name || doc.filename || doc.document_id;

        const meta = document.createElement("span");
        meta.className = "kb_doc_meta";
        meta.textContent = formatDocMeta(doc);

        item.append(name, meta);
        item.addEventListener("click", () => {
            window.location.hash = buildHash(doc.document_id, "");
        });
        docListEl.append(item);
    }
}

function renderHeader(doc, anchor) {
    titleEl.textContent = doc.doc_name || doc.filename || doc.document_id;
    const metaParts = [formatDocMeta(doc), "文档 ID " + doc.document_id];
    if (anchor) metaParts.push("锚点 " + anchor);
    metaEl.textContent = metaParts.join(" | ");
}

function renderEmpty(message) {
    chunkListEl.innerHTML = "";
    const empty = document.createElement("div");
    empty.className = "kb_empty";
    empty.textContent = message;
    chunkListEl.append(empty);
}

function renderError(message) {
    chunkListEl.innerHTML = "";
    const box = document.createElement("div");
    box.className = "kb_error";
    box.textContent = message;
    chunkListEl.append(box);
}

function buildChunkCard(chunk, isTarget) {
    const card = document.createElement("article");
    card.className = "chunk_card" + (isTarget ? " chunk_card--target" : "");
    card.dataset.anchor = chunk.anchor_id;
    card.dataset.index = String(chunk.chunk_index);

    const head = document.createElement("div");
    head.className = "chunk_head";

    const indexTag = document.createElement("span");
    indexTag.className = "chunk_index";
    indexTag.textContent = "#" + chunk.chunk_index;

    const meta = document.createElement("span");
    meta.className = "chunk_meta";
    const metaParts = [];
    if (chunk.page_number) metaParts.push("第 " + chunk.page_number + " 页");
    if (chunk.block_type && chunk.block_type !== "text") metaParts.push(chunk.block_type);
    metaParts.push((chunk.text || "").length + " 字");
    meta.textContent = metaParts.join(" · ");

    const anchorTag = document.createElement("code");
    anchorTag.className = "chunk_anchor";
    anchorTag.textContent = chunk.anchor_id;
    anchorTag.title = "点击复制锚点";
    anchorTag.addEventListener("click", async () => {
        try {
            await navigator.clipboard.writeText(chunk.anchor_id);
            anchorTag.textContent = "已复制";
            window.setTimeout(() => {
                anchorTag.textContent = chunk.anchor_id;
            }, 1200);
        } catch (error) {
            anchorTag.textContent = chunk.anchor_id;
        }
    });

    head.append(indexTag, meta, anchorTag);

    const text = document.createElement("p");
    text.className = "chunk_text";
    const raw = chunk.text || "";
    const needToggle = !isTarget && raw.length > MAX_PREVIEW;
    text.textContent = needToggle ? raw.slice(0, MAX_PREVIEW) + "…" : raw;

    card.append(head, text);

    if (needToggle) {
        const toggle = document.createElement("button");
        toggle.type = "button";
        toggle.className = "chunk_toggle";
        toggle.textContent = "展开全文";
        toggle.addEventListener("click", () => {
            const expanded = toggle.textContent === "收起";
            text.textContent = expanded ? raw.slice(0, MAX_PREVIEW) + "…" : raw;
            toggle.textContent = expanded ? "展开全文" : "收起";
        });
        card.append(toggle);
    }

    return card;
}

function renderChunks(chunks, anchor) {
    chunkListEl.innerHTML = "";
    if (!chunks.length) {
        renderEmpty("这份文档没有可展示的分块。");
        return;
    }

    const fragment = document.createDocumentFragment();
    for (const chunk of chunks) {
        fragment.append(buildChunkCard(chunk, chunk.anchor_id === anchor));
    }
    chunkListEl.append(fragment);

    if (!anchor) return;

    const target = Array.from(chunkListEl.children).find(
        (element) => element.dataset && element.dataset.anchor === anchor
    );
    if (!target) {
        const note = document.createElement("div");
        note.className = "kb_error";
        note.textContent = "锚点在当前文档中不存在（可能已重新入库）。左侧列表可切换其他文档。";
        chunkListEl.prepend(note);
        return;
    }
    target.scrollIntoView({ block: "center" });
}

async function resolveDocumentIdByAnchor(anchor) {
    try {
        const chunk = await requestJson("/api/knowledge/chunks?anchor_id=" + encodeURIComponent(anchor));
        return chunk.document_id || "";
    } catch (error) {
        return "";
    }
}

async function load() {
    const route = parseRoute();

    let docs;
    try {
        docs = await requestJson("/api/knowledge/documents");
    } catch (error) {
        statsEl.textContent = "加载失败";
        renderError("文档列表加载失败：" + error.message);
        return;
    }

    const totalChunks = docs.reduce((sum, doc) => sum + (doc.chunk_count || 0), 0);
    statsEl.textContent = docs.length + " 篇文档 · " + totalChunks + " 个分块";

    let activeDocId = route.doc;
    if (!activeDocId && route.anchor) {
        activeDocId = await resolveDocumentIdByAnchor(route.anchor);
    }
    if (!activeDocId && docs.length) {
        activeDocId = docs[0].document_id;
    }

    renderDocList(docs, activeDocId);

    if (!activeDocId) {
        titleEl.textContent = "知识库是空的";
        metaEl.textContent = "";
        renderEmpty("上传文档后即可在这里查看每一块的内容与锚点。");
        return;
    }

    const doc = docs.find((item) => item.document_id === activeDocId);
    if (!doc) {
        titleEl.textContent = "文档不存在";
        metaEl.textContent = "";
        renderError("找不到 document_id 为 " + activeDocId + " 的文档。它可能已被重新入库替换。");
        return;
    }

    renderHeader(doc, route.anchor);

    let chunks;
    try {
        chunks = await requestJson(
            "/api/knowledge/documents/" + encodeURIComponent(doc.document_id) + "/chunks"
        );
    } catch (error) {
        renderError("分块加载失败：" + error.message);
        return;
    }

    renderChunks(chunks, route.anchor);
}

window.addEventListener("hashchange", load);
load();

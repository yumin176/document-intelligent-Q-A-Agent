import re
from typing import Any

_FENCE_START_RE = re.compile(r'^( {0,3})(`{3,}|~{3,})(.*)$')
_FENCE_END_RE = re.compile(r'^( {0,3})(`{3,}|~{3,})[ \t]*$')
def split_code_regions(text: str) -> list[dict[str,Any]]:
    """只转换 Markdown 普通文本，保留 fenced code 和 inline code。"""
    """
    将文本分割为普通文本 / fenced code / inline code 片段。

    返回 list[dict]，每项：
      - {"type": "text",        "content": str}
      - {"type": "fenced_code", "content": str, "code": str, "lang": str, "closed": bool}
      - {"type": "inline_code", "content": str, "code": str}
    """
    parts: list[dict] = []
    lines = text.splitlines(keepends=True)
    n = len(lines)
    i = 0
    buf: list[str] = []

    def flush():
        if buf:
            parts.extend(_split_inline("".join(buf)))
            buf.clear()

    while i < n:
        raw = lines[i]
        stripped = raw.rstrip("\r\n")
        m = _FENCE_START_RE.match(stripped)

        # 反引号 fence 的 info string 不允许含反引号
        is_fence = bool(m) and not (m.group(2)[0] == "`" and "`" in m.group(3))

        if is_fence:
            indent, fence, info = m.groups()
            fence_char = fence[0]
            fence_len = len(fence)
            info = info.strip()

            flush()
            code_lines = [raw]
            i += 1
            closed = False

            while i < n:
                cur_raw = lines[i]
                cur = cur_raw.rstrip("\r\n")
                em = _FENCE_END_RE.match(cur)
                if (
                    em
                    and em.group(2)[0] == fence_char
                    and len(em.group(2)) >= fence_len
                ):
                    code_lines.append(cur_raw)
                    i += 1
                    closed = True
                    break
                code_lines.append(cur_raw)
                i += 1

            inner_lines = code_lines[1:-1] if closed else code_lines[1:]
            parts.append({
                "type": "fenced_code",
                "content": "".join(code_lines),
                "code": "".join(inner_lines),
                "lang": info.split()[0] if info else "",
                "closed": closed,
            })
        else:
            buf.append(raw)
            i += 1

    flush()
    return parts



def _split_inline(text: str) -> list[dict]:
    """把普通文本中的 inline code 切出来。inline code 不跨行。"""
    parts: list[dict] = []
    buf: list[str] = []
    i = 0
    n = len(text)

    while i < n:
        if text[i] == "`":
            # 统计开头连续反引号数
            j = i
            while j < n and text[j] == "`":
                j += 1
            run = j - i

            # 找正好 run 个反引号的结束位置，遇到换行就停
            k = j
            end_start = -1
            while k < n:
                if text[k] == "\n":
                    break
                if text[k] == "`":
                    m = k
                    while m < n and text[m] == "`":
                        m += 1
                    if m - k == run:
                        end_start = k
                        end = m
                        break
                    k = m
                else:
                    k += 1

            if end_start != -1:
                if buf:
                    parts.append({"type": "text", "content": "".join(buf)})
                    buf.clear()
                parts.append({
                    "type": "inline_code",
                    "content": text[i:end],
                    "code": text[j:end_start],
                })
                i = end
            else:
                buf.append(text[i:j])
                i = j
        else:
            buf.append(text[i])
            i += 1

    if buf:
        parts.append({"type": "text", "content": "".join(buf)})
    return parts

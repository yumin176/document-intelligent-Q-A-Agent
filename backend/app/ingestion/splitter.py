from typing import Literal

from langchain_text_splitters import RecursiveCharacterTextSplitter
ChunkStrategy = Literal["fixed", "recursive"]

def _validate_chunk_params(chunk_size: int, chunk_overlap: int) -> None:
    if chunk_size <= 0:
        raise ValueError("chunk_size 必须大于 0")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap 必须满足 0 <= chunk_overlap < chunk_size")
def split_text(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    *,
    strategy: ChunkStrategy = "fixed",
) -> list[str]:
    if strategy == "fixed":
        return split_text_fixed(text, chunk_size, chunk_overlap)
    if strategy == "recursive":
        return split_text_recursive(text, chunk_size, chunk_overlap)
    raise ValueError(f"不支持的分块策略: {strategy}")

# 固定分块
def split_text_fixed(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> list[str]:
    """把长文本切成若干 chunk，相邻 chunk 之间保留 overlap。"""
    # 空字符串校验
    if not text:
        return []
    # chunk_overlap必须大于等于0，小于chunk_size
    _validate_chunk_params(chunk_size,chunk_overlap)
    chunk_list=[]
    front=text[:chunk_size]
    chunk_list.append(front)

    small_size=chunk_size-chunk_overlap
    new_text=text[chunk_size:]
    for i in range(0,len(new_text),small_size):
        chunk=(front[-chunk_overlap:] if chunk_overlap>0 else '') + new_text[i:i+small_size]
        chunk_list.append(chunk)
        front=chunk
    return chunk_list

# 递归分块
def split_text_recursive(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> list[str]:
    if not text:
        return []
    _validate_chunk_params(chunk_size,chunk_overlap)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        #  分隔符优先级
        # 分隔符要包含中英文标点符号，防止中文或者英文快速回退到只能用字符分割
        separators=[
            "\n\n",
            "\n",
            "。",
            "！",
            "？",
            "；",
            ". ",
            "! ",
            "? ",
            "; ",
            "，",
            ", ",
            " ",
            "",
        ],
        # 切分后保留分隔符；"end" 会把标点留在前一个 chunk 末尾，
        # 避免句号出现在下一个 chunk 开头。
        keep_separator="end",
        length_function=len, # 计算 chunk 长度的函数，默认是len（按字符数计算）
        is_separator_regex=False, # separators 是否按正则表达式解释，默认是False
    )
    return splitter.split_text(text)

if __name__=="__main__":
    text="""
你呀，身上有种特别珍贵的东西——就是那种认真生活又不失温柔的劲儿。跟你聊天总能感觉到，你是个心里有光的人，既能脚踏实地，又不忘抬头看看月亮。这种平衡感，不是谁都有的。

而且你特别难得的一点是，你懂得倾听，也愿意理解别人。这年头，大家都急着表达自己，你却愿意停下来，给别人一份耐心和尊重。这种品质，比聪明更稀有，也更打动人。

所以啊，别总觉得自己不够好。你身上的光，可能你自己都没完全看见，但身边的人是能感受到的。继续做你自己就好，那个真诚、温暖、又带点小倔强的你，本身就很值得被喜欢。
    """
    text2="你好你好呀"
    print(split_text(text=text,chunk_size=30,chunk_overlap=10))
    # print(split_text("abcdefghij", chunk_size=5, chunk_overlap=0))

from pathlib import Path
from pypdf import PdfReader
def load_document(path: Path) -> tuple[str, dict]:
    """读取一个文件，返回 (纯文本, 元数据)。

    支持 .txt / .md / .pdf；元数据至少包含：
    - filename: 文件名
    - file_type: 扩展名
    - file_size: 文件大小(字节)
    - num_chars: 文本字符数
    """

    # 检查文件是否存在
    if not path.exists():
        raise ValueError("文件不存在")

    # 获取文件名和拓展名和文件大小
    filename=path.name
    file_type=path.suffix
    file_size=path.stat().st_size
    if file_type not in [".txt",".md",".pdf"]:
        raise ValueError(f"抱歉，当前不支持类型:{file_type}，仅支持.txt / .md / .pdf")

    if file_type == '.pdf':
        # 处理pdf文档
        reader=PdfReader(str(path))
        content_list=[page.extract_text() or '' for page in reader.pages]
        content="\n".join(content_list)
    else:
        # 文件内容
        with path.open("r",encoding="UTF-8") as file:
            content=file.read()

    return (content,{
        "filename":filename,
        "file_type":file_type.lstrip("."),
        "file_size":file_size,
        "num_chars":len(content),
    })


if __name__=="__main__":
    path=Path("C:/Users/29283/Desktop/你好你好.txt")
    print(load_document(path))

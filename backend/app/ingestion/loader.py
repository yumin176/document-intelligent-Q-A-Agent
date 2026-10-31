from dataclasses import dataclass
from pathlib import Path
from pypdf import PdfReader

@dataclass(frozen=True)
class ParsedPage:
    page_number: int
    text: str


@dataclass(frozen=True)
class ParsedDocument:
    pages: tuple[ParsedPage, ...]
    metadata: dict

    # @property用来把方法变成属性访问，直接使用.text就可以获取到text（）执行后的结果
    @property
    def text(self) -> str:
        return "\n\n".join(page.text.rstrip("\r\n") for page in self.pages)

def load_document(path: Path) -> tuple[str, dict]:
    parse_data=parse_document(path)
    content=parse_data.text
    metadata=parse_data.metadata
    return (content,{
        "filename":metadata.get("filename","未知来源"),
        "file_type":metadata.get("file_type",""),
        "file_size":metadata.get("file_size","未知大小"),
        "num_chars":len(content),
    })

def parse_document(path: Path) -> ParsedDocument:
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

    pages=[]
    if file_type == '.pdf':
        # 处理pdf文档
        reader=PdfReader(str(path))
        pages=[ParsedPage(page_number=i+1,text=page.extract_text() or '') for i,page in enumerate(reader.pages)]
    else:
        # 文件内容
        with path.open("r",encoding="UTF-8") as file:
            pages=[
                ParsedPage(page_number=1,text=file.read())
            ]

    return ParsedDocument(
        pages=tuple(pages),
        metadata={
            "filename":filename,
            "file_type":file_type.lstrip("."),
            "file_size":file_size,
            "page_count":len(pages),
        }
    )

if __name__=="__main__":
    # path=Path("C:/Users/29283/Desktop/你好你好.txt")
    # print(load_document(path))
    path=Path("E:/01-学习/06-我自己的语雀笔记/RAG语料库/_Coze_Drive_扣子_所有对话_主对话_agent学习工作台_项目_知识库语料_智答Agent知识库语料/上市公司年报/科大讯飞_2025年年度报告.pdf")
    content=parse_document(path)
    print(content.metadata)
    for page in content.pages[:5]:
        print(page.page_number)
        print(page.text)
        print("*"*30)

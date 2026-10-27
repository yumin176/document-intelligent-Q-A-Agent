
from abc import ABC,abstractmethod


class BaseCleaner(ABC):
    """所有清洗器的统一接口。"""
    name:str

    @abstractmethod
    def clean(self, text: str) -> str:
        """清洗文本并返回新文本，不应修改外部状态。"""

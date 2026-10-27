from collections.abc import Sequence
from dataclasses import dataclass
from hashlib import sha256

from app.ingestion.cleaning.base import BaseCleaner


def _sha256(text: str) -> str:
    return sha256(text.encode("utf-8")).hexdigest()
# frozen=True 一旦数据类实例被创建，你就不能再修改或删除它的属性。
@dataclass(frozen=True)
class CleanTraceStep:
    cleaner_name: str
    before_text: str
    after_text: str

@dataclass(frozen=True)
class CleanStep:
    cleaner_name: str
    before_chars: int
    after_chars: int
    before_sha256: str
    after_sha256: str


@dataclass(frozen=True)
class CleanResult:
    text: str
    #      容器类型  元素类型    省略号（表示"任意多个"）
    steps: tuple[CleanStep, ...] # 表示任意个多CleanStep组成的元组
    trace: tuple[CleanTraceStep, ...] = ()


class CleanerPipeline:
    def __init__(self, cleaners: Sequence[BaseCleaner]) -> None:
        self.cleaners = tuple(cleaners)

        names: set[str] = set()
        for cleaner in self.cleaners:
            if not isinstance(cleaner, BaseCleaner):
                raise TypeError("cleaners 中的元素必须实现 BaseCleaner")
            if not isinstance(cleaner.name, str) or not cleaner.name.strip():
                raise ValueError("cleaner 名称不能为空")
            if cleaner.name in names:
                raise ValueError(f"cleaner 名称重复: {cleaner.name}")
            names.add(cleaner.name)

    def clean(self, text: str, *,capture_trace: bool = False,) -> CleanResult:
        """
            默认 capture_trace=False，保持当前入库行为
            为 True 时记录每个 cleaner 执行前后的完整文本
        """
        if not isinstance(text, str):
            raise TypeError("text 必须是字符串")

        current = text
        steps: list[CleanStep] = []
        trace: list[CleanTraceStep]=[]
        for cleaner in self.cleaners:
            before = current
            current = cleaner.clean(current)
            if not isinstance(current, str):
                raise TypeError(f"清洗器 {cleaner.name} 返回了非字符串")

            steps.append(
                CleanStep(
                    cleaner_name=cleaner.name,
                    before_chars=len(before),
                    after_chars=len(current),
                    before_sha256=_sha256(before),
                    after_sha256=_sha256(current),
                )
            )
            if capture_trace:
                trace.append(
                    CleanTraceStep(
                        cleaner_name=cleaner.name,
                        before_text=before,
                        after_text=current
                    )
                )

        return CleanResult(text=current, steps=tuple(steps),trace=tuple(trace))

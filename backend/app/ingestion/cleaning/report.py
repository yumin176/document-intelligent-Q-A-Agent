from dataclasses import asdict
import difflib

from app.ingestion.cleaning.pipeline import CleanResult


def build_cleaning_report(
    source: str,
    raw_text: str,
    result: CleanResult,
    *,
    include_diff: bool = True,
    max_diff_chars: int = 12000,
) -> dict:
    if isinstance(max_diff_chars, bool) or not isinstance(max_diff_chars, int):
        raise TypeError("max_diff_chars 必须是整数")
    if max_diff_chars <= 0:
        raise ValueError("max_diff_chars 必须大于 0")
    # diff 需要每步的完整前后文本，因此必须显式开启 trace。
    if include_diff and len(result.trace) != len(result.steps):
        raise ValueError("生成 diff 报告需要先使用 capture_trace=True 执行清洗")

    merged = {}

    for item in result.steps + result.trace:
        name = item.cleaner_name
        if name in merged:
            merged[name].update(asdict(item))   # 相同 name，合并字段
        else:
            merged[name] = asdict(item)   # 第一次出现，复制一份
    steps=[]
    for item in list(merged.values()):
        before_chars= item["before_chars"]
        after_chars=item["after_chars"]
        delta_chars=after_chars - before_chars
        cleaner_name=item["cleaner_name"]
        # 字符数相同不代表内容没变，必须用哈希判断是否发生等长替换。
        changed = item["before_sha256"] != item["after_sha256"]
        diff = None
        if include_diff:
            before=item["before_text"]
            after=item["after_text"]
            # unified_diff方法输出文本格式的比较结果
            # unified_diff函数会返回一个生成器对象，该对象包含两个文本文件的差异
            # lineterm参数指定行结束符，fromfile和tofile参数分别指定要比较的两个文件名
            diff = "\n".join(difflib.unified_diff(
                before.splitlines(),
                after.splitlines(),
                fromfile=f"{cleaner_name}:before",
                tofile=f"{cleaner_name}:after",
                lineterm="",
            ))
            if len(diff) > max_diff_chars:
                diff = diff[:max_diff_chars] + "\n... [diff truncated]"
        steps.append({
            "cleaner": cleaner_name,
            "before_chars": before_chars,
            "after_chars": after_chars,
            "delta_chars": delta_chars,
            "before_sha256": item["before_sha256"],
            "after_sha256": item["after_sha256"],
            "changed": changed,
            "diff": diff,
        })

    return {
       "source":source,
       "raw_chars":len(raw_text),
       "cleaned_chars":len(result.text),
       "delta_chars":len(result.text)-len(raw_text),
       "steps":steps
    }
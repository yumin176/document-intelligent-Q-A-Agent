"""pytest 配置：确保能够 import 到 app 包，并隔离外部依赖。"""

import os
import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

# 强制关闭 LangSmith 追踪（env 优先级高于 .env）：
# 开发者本地 .env 打开追踪后，单测不该往远端发几百条 trace——
# 既污染项目，又会让测试变慢、不稳定。
os.environ["LANGSMITH_TRACING"] = "false"

from __future__ import annotations
import os 
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

def load_env(path :Path):
    if not path.exists():
        raise FileNotFoundError(" .env not found")
    
    for line in path.read_text(encoding = "utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or " = " not in line:
            continue
        key,_,value = line.partition(" = ")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key,value)


load_env(PROJECT_ROOT / ".env")

LLM_API_KEY = os.getenv("LLM_API_KEY","")
LLM_BASE_URL = os.getenv("LLM_BASE_URL","https://dashscope.aliyuncs.com/compatible-mode/v1")
LLM_MODEL = os.getenv("LLM_MODEL","qwen3.8-flash")
LLM_EXTAR_BODY: dict = json.loads(os.getenv("LLM_EXTAR_BODY","{}"))

# ---- Agent 运行时参数（loop.py / context.py 使用）----

# 一次 run() 里最多允许多少轮「LLM 调用 + 工具执行」，超过即兜底退出
MAX_TURNS = int(os.getenv("MAX_TURNS", "10"))

# 上下文 token 软阈值：估算超过它就触发早期对话压缩（摘要）
CONTEXT_TOKEN_LIMIT = int(os.getenv("CONTEXT_TOKEN_LIMIT", "6000"))

# 压缩时保留最近 N 个事件不压，保证模型记得刚发生的细节
KEEP_RECENT_EVENTS = int(os.getenv("KEEP_RECENT_EVENTS", "2"))

# 单条工具结果进入上下文前的最大字符数（超长截断，防撑爆窗口）
MAX_TOOL_RESULT_CHARS = int(os.getenv("MAX_TOOL_RESULT_CHARS", "4000"))

# 运行数据目录：会话 / 待办 / trace 日志都放在这里（默认随包走，已 gitignore）
DATA_DIR = Path(os.getenv("DATA_DIR", str(PROJECT_ROOT / "data")))
TRACE_PATH = DATA_DIR / "logs" / "trace.jsonl"


from __future__ import annotations

import json
import threading
from datetime import datetime
from pathlib import Path


def _now() -> str:
    """带毫秒的 ISO 时间戳——排查时序问题时毫秒级精度有用。"""
    return datetime.now().isoformat(timespec="milliseconds")


class Tracer:
    """把事件以 JSONL 追加写入文件（线程安全）。

    JSONL = 每行一个独立 JSON 对象。相比一个大 JSON 数组，追加写不需要
    读旧内容、天然适合日志场景（坏了也只坏最后一行）。
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()  

    def log(self, event: str, **fields) -> dict:
        """写一行事件并返回该记录（调用方顺便存进内存 trace）。"""
        record = {"ts": _now(), "event": event, **fields}
        line = json.dumps(record, ensure_ascii=False)
        with self._lock:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
        return record


class NullTracer:
    """空实现：不落盘，仅在内存里返回记录（单测用）。
    """

    def log(self, event: str, **fields) -> dict:
        return {"ts": _now(), "event": event, **fields}

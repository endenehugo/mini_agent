from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path





def _now() -> str:
    """当前时间的 ISO 字符串（精确到秒），用作会话的时间戳。"""
    return datetime.now().isoformat(timespec="seconds")


def _safe_name(session_id: str) -> str:
    """把 session_id 转成安全文件名：只保留字母数字和 -_，其余换成 _。

    防止 session_id 里出现 .. 或 / 等字符导致文件写到目录之外。
    """
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in session_id)


@dataclass
class Message:
    """一条对话消息，字段与 OpenAI 兼容协议的 wire 格式对齐，可直接回放给 API。

    四种角色（role）：
      user      —— 用户说的话；
      assistant —— 模型的回复；要调工具时 content 是思考文本，tool_calls 是调用列表；
      tool      —— 工具执行结果（靠 tool_call_id 与某个 tool_calls[id] 配对）；
      （system 提示词不存这里，构建上下文时由 ContextBuilder 动态加上。）
    """

    role: str                               
    content: str | None = None
    tool_calls: list[dict] | None = None     
    tool_call_id: str | None = None          
    name: str | None = None                  

    def to_dict(self) -> dict:
        """序列化成 API 用的 dict：只输出非 None 字段（协议不接受多余 null 字段）。"""
        d: dict = {"role": self.role}
        if self.content is not None:
            d["content"] = self.content
        if self.tool_calls is not None:
            d["tool_calls"] = self.tool_calls
        if self.tool_call_id is not None:
            d["tool_call_id"] = self.tool_call_id
        if self.name is not None:
            d["name"] = self.name
        return d

    @staticmethod
    def from_dict(d: dict) -> "Message":
        """从落盘 JSON 还原（反序列化）。"""
        return Message(
            role=d["role"],
            content=d.get("content"),
            tool_calls=d.get("tool_calls"),
            tool_call_id=d.get("tool_call_id"),
            name=d.get("name"),
        )



@dataclass
class Session:
    """一个聊天窗口的完整状态：消息历史 + 压缩产生的早期摘要。"""

    id: str
    messages: list[Message] = field(default_factory=list)
    summary: str = ""          # 压缩产生的早期对话摘要（构建上下文时放进 system prompt）
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)

    def append(self, msg: Message) -> None:
        """追加一条消息并刷新更新时间。"""
        self.messages.append(msg)
        self.updated_at = _now()

    def events(self) -> list[list[Message]]:
        """把消息切成「事件」：一个事件 = 一次用户输入引发的全部消息
        （user -> assistant(tool_calls) -> tool... -> assistant 最终回答）。

        压缩 / 裁剪只能按事件边界进行，保证 tool_call 与 tool 结果永不断裂配对
        （否则回放给 API 会直接报错）。
        """
        events: list[list[Message]] = []
        for msg in self.messages:
            if msg.role == "user" or not events:
                events.append([msg])
            else:
                events[-1].append(msg)
        return events



class SessionStore:
    """按 session_id 持久化到 data/sessions/<id>.json（原子写入：tmp + rename）。"""

    def __init__(self, data_dir: Path):
        self.dir = Path(data_dir) / "sessions"
        self.dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock() 

    def _path(self, session_id: str) -> Path:
        return self.dir / f"{_safe_name(session_id)}.json"

    def get_or_create(self, session_id: str) -> Session:
        """读会话；文件不存在（第一次用这个窗口）则返回空白会话。"""
        p = self._path(session_id)
        if p.exists():
            with self._lock, p.open("r", encoding="utf-8") as f:
                data = json.load(f)
            return Session(
                id=data["id"],
                messages=[Message.from_dict(m) for m in data.get("messages", [])],
                summary=data.get("summary", ""),
                created_at=data.get("created_at", _now()),
                updated_at=data.get("updated_at", _now()),
            )
        return Session(id=session_id)
    
    def save(self, session: Session) -> None:
        """全量落盘。原子写入：先写 .tmp 再 os.replace 顶替正式文件。

        为什么不直接写正式文件？如果写到一半断电/崩溃，会留下半截 JSON。
        先写临时文件再一步「重命名」，要么旧版本、要么新版本，绝不出现半截文件。
        """
        payload = {
            "id": session.id,
            "summary": session.summary,
            "created_at": session.created_at,
            "updated_at": session.updated_at,
            "messages": [m.to_dict() for m in session.messages],
        }
        with self._lock:
            p = self._path(session.id)
            tmp = p.with_suffix(".tmp")
            tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            os.replace(tmp, p)


    def list_sessions(self) -> list[str]:
        """按最近更新时间倒序返回所有 session id。"""
        paths = sorted(self.dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        return [p.stem for p in paths]

    def reset(self, session_id: str) -> bool:
        """删除该会话的落盘文件（= 清空窗口）。返回是否真的删了。"""
        p = self._path(session_id)
        if p.exists():
            p.unlink()
            return True
        return False
from __future__ import annotations

import json
import os
import threading
from pathlib import Path

from .base import Tool, ToolContext

# 允许的四种操作（写死成元组，供 handler 校验和 Schema 的 enum 复用）
_ACTIONS = ("add", "list", "complete", "delete")


class TodoStore:
    """session 级待办存储（JSON 文件，原子写入）。

    与 SessionStore 同款套路：一个 session 一个文件 + 线程锁 +
    tmp-then-rename 原子写入。区别是待办与对话历史分开存——
    清空聊天（reset session）不该误删待办，两者生命周期不同。
    """

    def __init__(self, data_dir: Path):
        self.dir = Path(data_dir) / "todos"
        self.dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def _path(self, session_id: str) -> Path:
        # 文件名安全化（同 SessionStore），防路径穿越
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in session_id)
        return self.dir / f"{safe}.json"

    def _load(self, session_id: str) -> list[dict]:
        """读取清单；没有文件返回空列表（第一次用）。"""
        p = self._path(session_id)
        if not p.exists():
            return []
        return json.loads(p.read_text(encoding="utf-8"))

    def _save(self, session_id: str, todos: list[dict]) -> None:
        """原子落盘：先写 .tmp 再替换，避免写一半留下坏 JSON。"""
        p = self._path(session_id)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(todos, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, p)

    def add(self, session_id: str, text: str) -> dict:
        """追加一条待办，id 自动递增（max+1），返回新增的条目。"""
        with self._lock:  # 锁住「读-改-写」整个区间，防并发下 id 重复
            todos = self._load(session_id)
            item = {"id": max((t["id"] for t in todos), default=0) + 1, "text": text, "done": False}
            todos.append(item)
            self._save(session_id, todos)
            return item

    def list(self, session_id: str) -> list[dict]:
        """列清单（只读，不需要加锁）。"""
        return self._load(session_id)

    def complete(self, session_id: str, todo_id: int) -> dict:
        """把指定 id 的待办标记为完成；找不到则抛 ValueError（由注册表转 ERROR 观测）。"""
        with self._lock:
            todos = self._load(session_id)
            for t in todos:
                if t["id"] == todo_id:
                    t["done"] = True
                    self._save(session_id, todos)
                    return t
            raise ValueError(f"待办 #{todo_id} 不存在")

    def delete(self, session_id: str, todo_id: int) -> dict:
        """删除指定 id 的待办；返回被删的条目。"""
        with self._lock:
            todos = self._load(session_id)
            for i, t in enumerate(todos):
                if t["id"] == todo_id:
                    self._save(session_id, todos[:i] + todos[i + 1:])  # 去掉第 i 条
                    return t
            raise ValueError(f"待办 #{todo_id} 不存在")


def _make_handler(store: TodoStore):
    """工厂函数：把 TodoStore「装进」handler 形成闭包，再返回 handler。

    为什么不直接写死一个模块级 handler？因为 store 是运行时按 data_dir
    构造的（见 tools/__init__.py），必须等它存在时再注入进来。
    """

    def _handler(args: dict, ctx: ToolContext) -> str:
        # ctx.session_id 是关键：所有操作只碰「当前窗口」的文件，
        # 这就是待办的会话隔离——不同窗口连编号都是从 1 各自开始的。
        action = args.get("action")
        if action == "add":
            text = str(args.get("text", "")).strip()
            if not text:
                raise ValueError("add 操作需要提供 text")
            item = store.add(ctx.session_id, text)
            # 返回文本刻意标注「仅当前会话可见」，帮模型理解隔离规则
            return f"已添加待办 #{item['id']}：{item['text']}（仅当前会话可见）"
        if action == "list":
            todos = store.list(ctx.session_id)
            if not todos:
                return "当前会话暂无待办。"
            # 列表格式化成人类可读的几行文本（模型转述给用户更自然）
            return "\n".join(f"#{t['id']} [{'已完成' if t['done'] else '未完成'}] {t['text']}" for t in todos)
        if action == "complete":
            item = store.complete(ctx.session_id, int(args.get("id", 0)))
            return f"待办 #{item['id']} 已标记完成：{item['text']}"
        if action == "delete":
            item = store.delete(ctx.session_id, int(args.get("id", 0)))
            return f"已删除待办 #{item['id']}：{item['text']}"
        # 传了不认识的 action：报错并把合法选项告诉模型，方便它纠正
        raise ValueError(f"action 必须是 {'/'.join(_ACTIONS)} 之一")

    return _handler


def make_todo_tool(store: TodoStore) -> Tool:
    """生成 todo 工具。注意 Schema 的设计细节：
    - action 用 enum 约束取值，模型想传错都难；
    - text/id 不放进 required（只有 action 必填），由 handler 按需校验——
      因为不同 action 需要的参数不同，交给 handler 精确报错比 Schema 死板
      约束更灵活。
    """
    return Tool(
        name="todo",
        description="管理当前会话的待办清单（按聊天窗口隔离）。支持 add/list/complete/delete 四种操作。",
        parameters={
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": list(_ACTIONS), "description": "操作类型"},
                "text": {"type": "string", "description": "add 时的待办内容"},
                "id": {"type": "integer", "description": "complete/delete 时的待办编号"},
            },
            "required": ["action"],
        },
        handler=_make_handler(store),
    )

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable


@dataclass
class ToolContext:
    """执行工具时的运行环境：当前是哪个会话、数据目录在哪。

    工具通过它实现「会话隔离」：比如 todo 工具拿到 session_id 后，
    只读写这个窗口自己的待办文件，不同窗口互不可见。
    """

    session_id: str
    data_dir: Path


@dataclass
class Tool:
    """一个工具的完整定义 = 元信息（给模型看） + 实现（给 Runtime 执行）。"""

    name: str
    description: str
    parameters: dict                             
    handler: Callable[[dict, ToolContext], str]   



class ToolRegistry:
    """工具注册表：Agent 持有它，主循环从这里取 Schema 和执行入口。"""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}  

    def register(self, tool: Tool) -> None:
        """注册工具：校验名字合法且不重复，防止运行时才发现问题。"""
        if not tool.name or " " in tool.name:
            raise ValueError(f"非法工具名: {tool.name!r}")
        if tool.name in self._tools:
            raise ValueError(f"工具重复注册: {tool.name}")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        """按名字取工具；不存在返回 None（调用方自行处理未知工具）。"""
        return self._tools.get(name)

    def names(self) -> list[str]:
        return list(self._tools)

    def openai_schemas(self) -> list[dict]:
        """转成 OpenAI function calling 的 tools 参数格式。

        即把每个 Tool 变成 {"type": "function", "function": {name, description, parameters}}，
        每次请求 LLM 时全量带上——模型「知道有哪些工具可用」靠的就是这个。
        """
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                },
            }
            for t in self._tools.values()
        ]

    def execute(self, name: str, arguments: dict, ctx: ToolContext) -> str:
        """执行工具；未知工具 / 参数缺失 / 内部异常 统一转成 ERROR 观测字符串。

        这是 Agent 不轻易崩掉的关键：无论工具怎么坏，主循环拿到的
        永远是一段可读文本，可以把它如实告诉模型/用户。
        """
        tool = self._tools.get(name)
        if tool is None:
            return f"ERROR: 未知工具 {name!r}，可用工具: {', '.join(self._tools)}"
        try:
            return str(tool.handler(arguments, ctx))
        except Exception as e:  
            return f"ERROR: {type(e).__name__}: {e}"

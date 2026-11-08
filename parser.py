from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from .llm import LLMResponse, ToolCall


_TEXT_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


@dataclass
class ParsedStep:
    """一次模型输出归一化后的结果：要么是工具调用轮，要么是最终答案轮。"""

    thought: str | None = None              
    tool_calls: list[ToolCall] = field(default_factory=list)
    final_answer: str | None = None         

    @property
    def is_tool_step(self) -> bool:
        """True = 模型要求调工具（本轮继续循环）；False = 出最终答案（本轮结束）。"""
        return bool(self.tool_calls)


def _parse_text_tool_calls(content: str) -> tuple[list[ToolCall], str]:
    """从正文解析文本协议工具调用，返回 (calls, 去除调用块后的正文)。

    用 re.sub 而非 findall：既取到了调用块，又顺手把它从正文里删除，
    剩下的正文就是干净的「思考过程」。
    """
    calls: list[ToolCall] = []

    def _make(match: re.Match) -> str:
        try:
            obj = json.loads(match.group(1))
        except json.JSONDecodeError:
            return match.group(0)  
        name = obj.get("name") or obj.get("tool")
        if not name:
            return match.group(0)
        args = obj.get("arguments", obj.get("args", obj.get("parameters", {})))
        if not isinstance(args, dict):
            args = {"value": args}  # 参数不是对象就包一层，保证 handler 拿到 dict
        calls.append(ToolCall(id=f"text_{len(calls) + 1}", name=str(name), arguments=args))
        return ""

    cleaned = _TEXT_TOOL_CALL_RE.sub(_make, content)
    return calls, cleaned.strip()

def parse_response(resp: LLMResponse) -> ParsedStep:
    """解析入口：优先结构化 tool_calls，其次文本协议兜底，最后按最终答案处理。"""
    content = (resp.content or "").strip()

    if resp.has_tool_calls:
        valid = [tc for tc in resp.tool_calls if tc.name]
        return ParsedStep(thought=content or None, tool_calls=valid)

    calls, cleaned = _parse_text_tool_calls(content)
    if calls:
        return ParsedStep(thought=cleaned or None, tool_calls=calls)

    return ParsedStep(thought=None, final_answer=content)
from __future__ import annotations

from .base import Tool, ToolContext

# mock 知识库：关键词 -> 词条内容。真实场景这里是「调搜索 API」。
_KB = {
    "python": "Python 是一门解释型高级编程语言，由 Guido van Rossum 于 1991 年首次发布。",
    "agent": "AI Agent（智能体）= LLM + 记忆 + 工具调用 + 规划，能在「思考-行动-观察」循环中自主完成任务。",
    "react": "ReAct 模式：Reason 与 Act 交替进行，模型先思考再调用工具，观察结果后继续推理，直到得出答案。",
    "function calling": "Function Calling 让 LLM 输出结构化的函数调用请求（名称 + JSON 参数），"
    "由外部程序执行后把结果回填给模型继续推理。",
    "mcp": "MCP（Model Context Protocol）是 Anthropic 提出的开放协议，用于统一模型与外部工具/数据源的连接方式。",
    "todo": "待办清单（todo list）是把长程任务外化的常用手段，能显著缓解长上下文中的目标漂移。",
}


def _search(args: dict, ctx: ToolContext) -> str:
    """模拟一次搜索：query 里出现词条关键词（或其分词）即算命中。"""
    query = str(args.get("query", "")).strip()
    if not query:
        raise ValueError("query 不能为空")
    q = query.lower()
    hits = [(k, v) for k, v in _KB.items() if k in q or any(w in q for w in k.split())]
    if not hits:
        return (
            f"没有找到与「{query}」相关的结果。"
            f"（注意：这是 mock 搜索，知识库仅包含 {list(_KB)} 等词条，请如实告诉用户查不到。）"
        )
    lines = [f"{i}. [{k}] {v}" for i, (k, v) in enumerate(hits[:2], 1)]
    return "搜索结果（mock）:\n" + "\n".join(lines)


SEARCH_TOOL = Tool(
    name="search",
    description="搜索资料（当前为 mock 实现，知识库有限）。用于查询事实性、时效性信息；查不到时要如实告知用户。",
    parameters={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "搜索关键词或问题"},
        },
        "required": ["query"],
    },
    handler=_search,
)

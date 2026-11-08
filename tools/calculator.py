from __future__ import annotations

import ast
import math
import operator

from .base import Tool, ToolContext

# 二元运算符白名单：ast 节点类型 -> 对应的 Python 函数
_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
# 一元运算符白名单：+x 和 -x
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}
# 允许的常量：注意 ast.Name 节点只认这两个名字
_CONSTS = {"pi": math.pi, "e": math.e}
# 允许的函数：全部要求单参数、不允许关键字传参
_FUNCS = {
    "sqrt": math.sqrt,
    "abs": abs,
    "log": math.log,
    "log2": math.log2,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
}


def _eval_node(node: ast.AST):
    """递归求值一棵 ast 子树。不认识的节点类型一律 raise —— 白名单语义。

    即：不是「屏蔽已知的危险操作」，而是「只放行已知的安全操作」，
    前者永远堵不完（例如属性访问、下标、lambda 都能藏代码），后者天然安全。
    """
    # 数字字面量（显式排除 bool：True/False 是 int 的子类，不希望参与运算）
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return node.value
    # 二元运算：左递归 + 右递归
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        return _BIN_OPS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    # 一元运算：只作用于一个操作数（如 -5）
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_eval_node(node.operand))
    # 常量名（pi / e）
    if isinstance(node, ast.Name) and node.id in _CONSTS:
        return _CONSTS[node.id]
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in _FUNCS
        and len(node.args) == 1
        and not node.keywords
    ):
        return _FUNCS[node.func.id](_eval_node(node.args[0]))
    raise ValueError(f"不支持的表达式元素: {type(node).__name__}")


def _calculate(args: dict, ctx: ToolContext) -> str:
    """工具入口：args 是模型给的参数 dict，返回值是给模型看的观测文本。"""
    expression = str(args.get("expression", "")).strip()
    if not expression:
        raise ValueError("expression 不能为空")
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as e:
        raise ValueError(f"表达式语法错误: {e.msg}") from e  
    result = _eval_node(tree.body)
    if isinstance(result, float):
        if result.is_integer() and abs(result) < 1e15:
            result = int(result)
        else:
            result = round(result, 10)
    return str(result)


CALCULATOR_TOOL = Tool(
    name="calculator",

    description="精确计算数学表达式。支持 + - * / // % **、括号、sqrt/log/log2/sin/cos/tan/abs、常量 pi/e。"
    "涉及任何数字计算时必须使用本工具，不要心算。",
    parameters={
        "type": "object",
        "properties": {
            "expression": {"type": "string", "description": "数学表达式，例如 (3+4)*12/2 或 sqrt(2)"},
        },
        "required": ["expression"],
    },
    handler=_calculate,
)

"""内置工具包。build_default_registry 返回装好四个内置工具的注册表。"""
from .base import Tool, ToolContext, ToolRegistry
from .calculator import CALCULATOR_TOOL
from .search import SEARCH_TOOL
from .todo import TodoStore, make_todo_tool
from .weather import WEATHER_TOOL


def build_default_registry(data_dir) -> ToolRegistry:
    """组装默认工具集。TodoStore 依赖 data_dir，所以走工厂函数注入。"""
    registry = ToolRegistry()
    registry.register(CALCULATOR_TOOL)
    registry.register(SEARCH_TOOL)
    registry.register(WEATHER_TOOL)
    registry.register(make_todo_tool(TodoStore(data_dir)))
    return registry


__all__ = [
    "Tool",
    "ToolContext",
    "ToolRegistry",
    "build_default_registry",
]

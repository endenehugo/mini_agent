from __future__ import annotations

from datetime import date

from .base import Tool, ToolContext

# 伪随机可选的天气状况
_CONDITIONS = ["晴", "多云", "阴", "小雨", "雷阵雨"]


def _weather(args: dict, ctx: ToolContext) -> str:
    city = str(args.get("city", "")).strip()
    if not city:
        raise ValueError("city 不能为空")
    seed = sum(ord(c) for c in city) + date.today().toordinal()
    cond = _CONDITIONS[seed % len(_CONDITIONS)]   
    low = 8 + seed % 12                           
    high = low + 6 + seed % 8                    
    return f"{city} 今日（{date.today().isoformat()}）天气（mock）：{cond}，{low}~{high}°C，微风，空气质量良。"


WEATHER_TOOL = Tool(
    name="weather",
    description="查询指定城市今天的天气（当前为 mock 数据）。",
    parameters={
        "type": "object",
        "properties": {
            "city": {"type": "string", "description": "城市名，例如 杭州"},
        },
        "required": ["city"],
    },
    handler=_weather,
)

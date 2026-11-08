"""mini_agent —— 一个最小可用的自研 Agent Runtime。

模块分工：
  config.py     —— 读配置（.env、模型名、轮数上限、上下文阈值等）
  llm.py        —— 薄 LLM 客户端：发消息 -> 拿结构化响应
  parser.py     —— 解析模型输出：思考 / 工具调用 / 最终答案
  loop.py       —— Agent 主循环（ReAct：思考-行动-观察）
  session.py    —— 会话持久化（一个聊天窗口一个 JSON 文件）
  context.py    —— 上下文组装与压缩（摘要 + 硬截断）
  tools/        —— 工具注册表与内置工具（calculator/search/weather/todo）
  tracing.py    —— JSONL 事件日志
"""

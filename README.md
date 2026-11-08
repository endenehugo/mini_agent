# mini_agent

一个最小可用、**零框架自研**的 AI Agent Runtime（约 800 行 Python，不依赖 langgraph / langchain / openhands 等任何 Agent 框架）。

核心是一个 ReAct 主循环：**思考 → 行动（调用工具）→ 观察（工具结果回填）→ 再思考**，直到模型给出最终答案。支持多会话隔离、会话持久化、上下文自动压缩、工具调用全链路 trace 日志。

## 功能特性

| 能力 | 说明 |
| --- | --- |
| Agent 主循环 | 自研 Runtime（`loop.py`），支持并行多工具调用、最大轮数兜底 |
| 工具系统 | 注册表模式（`tools/base.py`）：模型看 Schema 决策，Runtime 负责执行；工具崩溃自动转 ERROR 观测文本，主流程不中断 |
| 内置工具 | `calculator`（白名单 AST 求值，防代码注入）/ `search`（mock）/ `weather`（mock）/ `todo`（按会话隔离的待办清单） |
| 会话管理 | 一个聊天窗口一个 JSON 文件（`session.py`），原子写入（tmp + rename），重启不丢 |
| 上下文压缩 | 超阈值先把早期对话 LLM 摘要压缩，仍超限再按「完整事件」硬截断（`context.py`），绝不拆散 tool_call 与结果的配对 |
| 可观测性 | 每一步事件（思考/调用/耗时/token）落盘 JSONL（`tracing.py`） |
| 文本协议兜底 | 模型不吐结构化 tool_calls 时，解析 `<tool_call>{...}</tool_call>` 文本协议（`parser.py`） |

## 项目结构

```text
mini_agent/
├── config.py          # 读 .env、Agent 运行参数（MAX_TURNS / 上下文阈值等）
├── llm.py             # 薄 LLM 客户端：OpenAI 兼容协议 -> 结构化 LLMResponse
├── parser.py          # 解析模型输出：思考 / 工具调用 / 最终答案
├── loop.py            # ★ Agent 主循环（ReAct 发动机）
├── session.py         # 会话持久化：消息历史 + 事件切分
├── context.py         # 每轮组装上下文：token 估算、摘要压缩、硬截断
├── tracing.py         # JSONL 事件日志
├── tools/
│   ├── base.py        # Tool / ToolContext / ToolRegistry
│   ├── calculator.py  # 白名单 AST 数学求值（防注入）
│   ├── search.py      # mock 搜索
│   ├── weather.py     # mock 天气
│   └── todo.py        # 会话隔离待办（JSON 原子落盘）
└── tests/
    ├── test_units.py  # 51 个单元测试（不联网）
    └── e2e_manual.py  # 端到端冒烟测试（需真实 API Key）
```

## 快速开始

### 1. 安装依赖

```bash
pip install openai>=2.0 pytest
```

要求 Python 3.8+（代码用了 `from __future__ import annotations`，类型标注兼容 3.8）。

### 2. 配置密钥（敏感信息不入库）

`.env` 存放真实密钥，已被 `.gitignore` 忽略；仓库只保留模板 `.env.example`：

```bash
# Windows PowerShell
Copy-Item .env.example .env
# Linux / macOS
cp .env.example .env
```

然后编辑 `.env`，填入你自己的 Key：

```ini
LLM_BASE_URL = https://dashscope.aliyuncs.com/compatible-mode/v1
LLM_API_KEY  = sk-你的真实Key
LLM_MODEL    = qwen3.8-flash
```

> ⚠️ 安全约定：任何真实 Key / token / 凭据只放 `.env`，不要写进代码或 README；`.gitignore` 已拦截 `.env`、`*.key`、`*.pem`、`credentials*.json` 等文件。提交前可用 `git status` 确认 `.env` 不在待提交列表里。

### 3. 运行 CLI

> 需在 `mini_agent` 的**上一级目录**以模块方式启动（内部使用包相对导入）。

```bash
python -m mini_agent.cli                 # 默认窗口 default
python -m mini_agent.cli window1         # 窗口1（独立会话，重启不丢）
python -m mini_agent.cli window2         # 另一个终端 => 完全独立的会话
python -m mini_agent.cli --list          # 列出所有会话
python -m mini_agent.cli window1 --reset # 清空该窗口
```

会话内命令：`/sessions` 列会话、`/reset` 清空当前会话、`/exit` 退出。

运行示例：

```text
mini-agent | 模型: qwen3.8-flash | 会话: window1 | 历史 0 条消息
命令: /sessions /reset /exit

你> 北京今天的天气怎么样？
  🔧 weather({"city": "北京"})

Agent> 北京今天（2026-11-08）天气：雷阵雨，15~24°C，微风，空气质量良。
  (final, 2 轮; trace: ...\mini_agent\data\logs\trace.jsonl)
```

运行产生的数据（会话 / 待办 / trace）都在 `data/` 目录下，已被 gitignore，不随仓库分发。

## 运行测试

```bash
# 在 mini_agent 的上一级目录执行
python -m pytest mini_agent/tests -v        # 51 个单元测试，不联网、不耗 token

# 端到端冒烟测试（真实调用 LLM API，需要 .env 配好 Key）
# Windows PowerShell:
$env:PYTHONPATH = "<项目上一级目录的绝对路径>"; python mini_agent\tests\e2e_manual.py
```

单元测试覆盖：

- **calculator**：11 组合法表达式 + 10 组注入/非法输入（`__import__`、`open`、lambda、赋值等全部拒绝）
- **todo**：增删改查、会话隔离、重启后 id 接续、路径穿越防护
- **ToolRegistry**：重复注册/非法名拒绝、未知工具与 handler 异常统一转 ERROR 观测
- **Session**：消息序列化 roundtrip、按 user 消息切分事件、持久化与 reset
- **parser**：结构化 tool_calls、`<tool_call>` 文本协议兜底、坏 JSON 不误判
- **ContextBuilder**：token 估算、软压缩合并摘要、压缩失败时硬截断且不拆散工具配对

## 核心设计要点

1. **白名单而非黑名单**：calculator 不「屏蔽危险操作」，而是「只放行已知安全的 AST 节点」，天然免疫代码注入。
2. **工具永不崩掉主循环**：`ToolRegistry.execute` 把任何异常转成 `ERROR: ...` 观测文本回给模型，模型可以下一轮自我纠正。
3. **事件级上下文管理**：压缩/截断的最小单位是「一次用户输入引发的完整事件」，保证 `assistant.tool_calls` 与 `tool` 结果永远成对，回放给 API 不会报错。
4. **两级容错**：LLM 摘要压缩失败 → 保守跳过；仍超限 → 按事件硬截断兜底。压缩是锦上添花，绝不因它中断主流程。
5. **原子写入**：会话与待办落盘均为「写 .tmp → os.replace」，崩溃不会留下半截 JSON。

## License

MIT

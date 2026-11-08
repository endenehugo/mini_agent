"""Agent 主循环 —— 核心 Runtime（自研，未使用 langgraph / openhands 等任何 Agent 框架）。

【零基础导览：这个文件在做什么？】

普通 LLM 对话只能「一问一答」；Agent（智能体）的区别在于：模型可以在回答
过程中主动要求「调用工具」（计算器 / 搜索 / 天气 / 待办…），拿到工具返回
的真实结果后再继续推理，直到能给出最终答案。这个「思考 -> 行动 -> 观察 ->
再思考」的循环就是 Agent 的核心工作方式（业内称 ReAct 模式）。

本模块就是整个 Runtime 的「发动机」，一次 run() 对应题目要求的 Loop：
  1. 接收用户输入，写入 session 并持久化；
  2. LLM 基于工具 Schema（工具说明书）决策：直接回答 or 调用工具（可并行多个）；
  3. 执行工具，把结果作为 observation（观测）回填 session；
  4. 回到 2 继续循环，直到 LLM 给出最终答案 / 达到最大轮次兜底退出。

各依赖模块的分工（建议按此顺序阅读）：
  config.py     —— 读配置（模型名、轮数上限、上下文阈值等）；
  llm.py        —— 薄薄的 LLM 客户端：发消息 -> 拿结构化响应，仅此而已；
  parser.py     —— 从 LLM 响应里解析出「思考 / 工具调用 / 最终答案」；
  session.py    —— 会话持久化：每个聊天窗口一个 JSON 文件，重启不丢；
  context.py    —— 每轮调用 LLM 前组装上下文，太长时自动压缩成摘要；
  tools/base.py —— 工具注册表：模型看 Schema 决策，Runtime 负责执行；
  tracing.py   —— 把每一步写成 JSONL 日志，方便回放排查（不进上下文）。
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from . import config
from .context import ContextBuilder
from .llm import LLMClient, ToolCall
from .parser import parse_response
from .session import Message, SessionStore
from .tools.base import ToolContext, ToolRegistry
from .tracing import Tracer

DEFAULT_SYSTEM_PROMPT = """你是一个小型任务型 Agent，可通过工具完成计算、资料查询、天气查询和待办管理。

决策规则：
1. 闲聊、解释、观点类问题：直接回答，不要调用工具。
2. 精确计算用 calculator；查资料用 search；查天气用 weather；待办的增删改查用 todo。
3. 工具可能失败：失败时向用户如实说明或换一种方式，禁止编造工具结果。
4. 待办按当前会话隔离（不同窗口互不可见）。
5. 用中文简洁回答。"""


@dataclass
class AgentResult:
    """一次 run() 的返回值：给调用方（如 CLI）看的执行摘要。

    trace 是本次运行的完整事件流水（思考了什么、调了什么工具、耗时多少），
    CLI 拿它来打印中间过程；logs/trace.jsonl 里还有一份持久化版本。
    """

    final_answer: str | None
    stop_reason: str            
    turns: int                  
    trace: list[dict] = field(default_factory=list)


def _clip(s: str, limit: int) -> str:
    """把超长字符串截断到 limit 个字符（防止撑爆上下文窗口）。"""
    return s if len(s) <= limit else s[:limit] + "…[已截断]"


def _dumps_args(arguments: dict | None) -> str:
    """参数落盘为 wire 格式字符串；解析失败时用 {} 占位，观测结果会提示模型纠正。

    背景：OpenAI 协议里，assistant 消息上的 tool_calls 是「字符串化的 JSON 参数」，
    这里的 wire 格式指协议线上传输的格式，序列化成字符串是为了能直接回放给 API。
    """
    return json.dumps(arguments, ensure_ascii=False) if arguments is not None else "{}"


class Agent:
    """Agent Runtime：持有 LLM 客户端、工具注册表、会话存储，对外只暴露 run()。"""

    def __init__(
        self,
        llm: LLMClient,
        registry: ToolRegistry,
        store: SessionStore,
        *,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        max_turns: int | None = None,
        token_limit: int | None = None,
        keep_recent_events: int | None = None,
        data_dir: Path | None = None,
        tracer: Tracer | None = None,
    ):
        
        self.llm = llm
        self.registry = registry
        self.store = store
        self.max_turns = max_turns or config.MAX_TURNS         
        self.data_dir = data_dir or config.DATA_DIR            
        self.context = ContextBuilder(
            llm,
            system_prompt,
            token_limit or config.CONTEXT_TOKEN_LIMIT,         
            keep_recent_events or config.KEEP_RECENT_EVENTS,    
        )
        self.tracer = tracer  

    # ------------------------------------------------------------------

    def run(self, session_id: str, user_input: str) -> AgentResult:
        """执行一轮完整任务：从用户输入出发，循环调用 LLM/工具直到得出最终答案。"""
        trace: list[dict] = []

        def rec(event: str, **fields) -> None:
            # 内部小工具函数：每个关键事件既写进 trace 列表（给本次返回值），
            # 也通过 tracer 落盘成 JSONL（给事后排查）。事件名即「发生了什么」。
            record = self.tracer.log(event, session=session_id, **fields) if self.tracer else {"event": event, **fields}
            trace.append(record)

        # ---- Step 1：用户输入进会话并立刻持久化 ----
        # get_or_create：window1 第一次用则新建，之后重启进程也能从磁盘接着聊。
        session = self.store.get_or_create(session_id)
        session.append(Message(role="user", content=user_input))
        self.store.save(session)
        rec("user_input", content=user_input)

        # ---- Step 2~4：主循环，每轮 = 一次 LLM 调用 ----
        for turn in range(1, self.max_turns + 1):
            # Step 2 前置：构建上下文（超阈值自动压缩早期对话）
            # build() 返回的 messages 是最终发给 API 的完整消息列表。
            messages, ctx_info = self.context.build(session)
            if ctx_info.get("compressed") or ctx_info.get("hard_truncated"):
                rec("context_changed", **ctx_info)  # 压缩/裁剪也是重要事件，记一笔
            rec("llm_request", turn=turn, n_messages=len(messages), est_tokens=ctx_info["est_tokens"])

            t0 = time.perf_counter()  # 计时起点，用于统计本次 LLM 调用延迟
            try:
                # 真正调用 LLM。tools= 注册表转出的 OpenAI Schema 列表，
                # 模型据此知道「有哪些工具可调、参数长什么样」。
                resp = self.llm.chat(messages, tools=self.registry.openai_schemas() or None)
            except Exception as e:  # SDK 已内置重试，到这里是最终失败（网络/鉴权/限流等）
                rec("llm_error", turn=turn, error=f"{type(e).__name__}: {e}")
                return AgentResult(None, "llm_error", turn, trace)
            latency_ms = round((time.perf_counter() - t0) * 1000)

            # 解析本轮输出：思考过程 / 工具调用 / 最终答案
            step = parse_response(resp)
            rec(
                "llm_response",
                turn=turn,
                latency_ms=latency_ms,
                finish_reason=resp.finish_reason,
                thought=step.thought,
                n_tool_calls=len(step.tool_calls),
                usage=resp.usage,
            )

            if step.is_tool_step:
                # ---- Step 3：模型要求调用工具 ----
                # 3a. 先把「模型发出了工具调用」这件事本身写入会话（assistant 消息）。
                #     这一步必不可少：OpenAI 协议要求 tool 消息必须紧跟在带
                #     tool_calls 的 assistant 消息之后，否则回放历史时 API 会报错。
                session.append(
                    Message(
                        role="assistant",
                        content=step.thought,
                        tool_calls=[
                            {
                                "id": tc.id,            # 调用唯一标识，用于和结果配对
                                "type": "function",
                                "function": {"name": tc.name, "arguments": _dumps_args(tc.arguments)},
                            }
                            for tc in step.tool_calls
                        ],
                    )
                )
                # 3b. 逐个执行工具，每个结果作为 role="tool" 的消息回填会话。
                for tc in step.tool_calls:
                    obs = self._execute_tool(session_id, tc, rec)
                    session.append(Message(role="tool", content=obs, tool_call_id=tc.id, name=tc.name))
                self.store.save(session)
                # ---- Step 4：带着工具结果回到循环开头，让 LLM 决定继续调工具还是给答案 ----
                continue

            # 没有工具调用 => 模型认为可以收尾，这就是最终答案
            final = (step.final_answer or "").strip() or "（模型返回了空回答）"
            session.append(Message(role="assistant", content=final))
            self.store.save(session)
            rec("final_answer", turn=turn, content=final)
            return AgentResult(final, "final", turn, trace)
        fallback = f"已连续调用工具 {self.max_turns} 轮仍未得出结论，本轮任务中止。请把任务拆小后重试。"
        session.append(Message(role="assistant", content=fallback))
        self.store.save(session)
        rec("max_turns_reached", max_turns=self.max_turns)
        return AgentResult(fallback, "max_turns", self.max_turns, trace)

    # ------------------------------------------------------------------

    def _execute_tool(self, session_id: str, tc: ToolCall, rec) -> str:
        """执行单个工具调用，返回 observation 字符串（即工具结果文本）。

        注意返回值是给模型看的：参数坏了也要返回一段可读的 ERROR 描述，
        让模型有机会在下一轮自我纠正，而不是让整个程序崩掉。
        """
        if tc.arguments is None:
            rec("tool_error", tool=tc.name, error="arguments 不是合法 JSON")
            return f"ERROR: 工具 {tc.name} 的参数不是合法 JSON，请修正后重试。"
        t0 = time.perf_counter()
        obs = self.registry.execute(tc.name, tc.arguments, ToolContext(session_id=session_id, data_dir=self.data_dir))
        rec(
            "tool_call",
            tool=tc.name,
            arguments=tc.arguments,
            latency_ms=round((time.perf_counter() - t0) * 1000),
            result=_clip(obs, 2000),
        )
        # 超长工具结果截断后再进上下文，防止撑爆窗口
        return _clip(obs, config.MAX_TOOL_RESULT_CHARS)

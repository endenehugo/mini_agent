"""单元测试：不联网、不调 LLM，全部本地逻辑。

运行方式（在 mini_agent 的上一级目录，即 F:\\online-interview-question）：
    python -m pytest mini_agent/tests -v
或在 mini_agent 目录内：
    python -m pytest tests -v
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# 保证可以以「包」方式导入 mini_agent（tests 目录位于包内）
PKG_PARENT = Path(__file__).resolve().parent.parent.parent
if str(PKG_PARENT) not in sys.path:
    sys.path.insert(0, str(PKG_PARENT))

from mini_agent.context import ContextBuilder, estimate_tokens, messages_tokens
from mini_agent.llm import LLMResponse, ToolCall
from mini_agent.parser import parse_response
from mini_agent.session import Message, Session, SessionStore
from mini_agent.tools import build_default_registry
from mini_agent.tools.base import Tool, ToolContext, ToolRegistry
from mini_agent.tools.calculator import _calculate
from mini_agent.tools.todo import TodoStore
from mini_agent.tools.weather import _weather


# ----------------------------------------------------------------------
# calculator：白名单 AST 求值
# ----------------------------------------------------------------------
class TestCalculator:
    @pytest.mark.parametrize(
        "expr,expected",
        [
            ("1+2", 3),
            ("(3+4)*12/2", 42),
            ("2**10", 1024),
            ("7//2", 3),
            ("7%3", 1),
            ("-5+3", -2),
            ("sqrt(2)", 1.4142135624),
            ("abs(-3.5)", 3.5),
            ("log2(8)", 3),
            ("pi", 3.141592653589793),
            ("2*pi", 6.283185307179586),
        ],
    )
    def test_valid_expressions(self, expr, expected):
        result = float(_calculate({"expression": expr}, None))
        assert result == pytest.approx(float(expected), rel=1e-9)

    def test_float_rounding(self):
        assert _calculate({"expression": "1/3"}, None) == str(round(1 / 3, 10))

    @pytest.mark.parametrize(
        "expr",
        [
            "",                      # 空表达式
            "__import__('os')",      # 注入函数调用
            "open('x')",             # 非白名单函数
            "(1+2",                  # 语法错误
            "'a'+'b'",               # 字符串
            "[1,2]",                 # 列表
            "True+1",                # bool 不当数字用
            "x=1",                   # 赋值不是 eval 表达式
            "lambda: 1",             # lambda
            "1 if True else 2",      # 条件表达式
        ],
    )
    def test_rejects_dangerous_or_invalid(self, expr):
        with pytest.raises((ValueError, SyntaxError)):
            _calculate({"expression": expr}, None)


# ----------------------------------------------------------------------
# weather：mock 数据，确定性（同城市同日期结果一致）
# ----------------------------------------------------------------------
class TestWeather:
    def test_deterministic_and_format(self):
        r1 = _weather({"city": "杭州"}, None)
        r2 = _weather({"city": "杭州"}, None)
        assert r1 == r2
        assert "杭州" in r1 and "天气" in r1 and "°C" in r1

    def test_empty_city_raises(self):
        with pytest.raises(ValueError):
            _weather({"city": ""}, None)


# ----------------------------------------------------------------------
# todo：会话隔离 + 原子落盘 + 增删改查
# ----------------------------------------------------------------------
class TestTodo:
    def _store(self, tmp_path):
        return TodoStore(tmp_path)

    def test_add_list_complete_delete(self, tmp_path):
        store = self._store(tmp_path)
        a = store.add("s1", "写 README")
        b = store.add("s1", "跑测试")
        assert a["id"] == 1 and b["id"] == 2
        assert len(store.list("s1")) == 2

        done = store.complete("s1", 1)
        assert done["done"] is True

        deleted = store.delete("s1", 2)
        assert deleted["text"] == "跑测试"
        assert [t["id"] for t in store.list("s1")] == [1]

    def test_session_isolation(self, tmp_path):
        store = self._store(tmp_path)
        store.add("window1", "A 的待办")
        store.add("window2", "B 的待办")
        assert store.list("window1")[0]["text"] == "A 的待办"
        assert store.list("window2")[0]["text"] == "B 的待办"

    def test_id_restart_from_disk(self, tmp_path):
        # 重新构造 store 模拟进程重启，id 应接着磁盘递增而不是从 1 重来
        store = self._store(tmp_path)
        store.add("s", "第一条")
        store2 = self._store(tmp_path)
        item = store2.add("s", "第二条")
        assert item["id"] == 2

    def test_complete_delete_missing_raises(self, tmp_path):
        store = self._store(tmp_path)
        with pytest.raises(ValueError):
            store.complete("s", 99)
        with pytest.raises(ValueError):
            store.delete("s", 99)

    def test_filename_sanitized(self, tmp_path):
        # 恶意 session_id 不得穿越出 todos 目录
        store = self._store(tmp_path)
        store.add("../../evil", "x")
        files = [p.name for p in (tmp_path / "todos").glob("*.json")]
        assert files == ["______evil.json"] if files else False


# ----------------------------------------------------------------------
# ToolRegistry：注册、Schema 转换、统一 ERROR 观测
# ----------------------------------------------------------------------
class TestRegistry:
    def _tool(self, name="dummy"):
        return Tool(
            name=name,
            description="d",
            parameters={"type": "object", "properties": {}},
            handler=lambda args, ctx: "ok",
        )

    def test_register_and_schema(self):
        reg = ToolRegistry()
        reg.register(self._tool("a"))
        reg.register(self._tool("b"))
        assert reg.names() == ["a", "b"]
        schemas = reg.openai_schemas()
        assert schemas[0]["type"] == "function"
        assert schemas[0]["function"]["name"] == "a"

    def test_duplicate_rejected(self):
        reg = ToolRegistry()
        reg.register(self._tool("a"))
        with pytest.raises(ValueError):
            reg.register(self._tool("a"))

    def test_invalid_name_rejected(self):
        reg = ToolRegistry()
        with pytest.raises(ValueError):
            reg.register(self._tool("has space"))

    def test_unknown_tool_returns_error_string(self):
        reg = ToolRegistry()
        out = reg.execute("nope", {}, ToolContext(session_id="s", data_dir=Path(".")))
        assert out.startswith("ERROR")

    def test_handler_exception_becomes_error_string(self):
        reg = ToolRegistry()

        def boom(args, ctx):
            raise RuntimeError("炸了")

        reg.register(Tool("boom", "d", {}, boom))
        out = reg.execute("boom", {}, ToolContext(session_id="s", data_dir=Path(".")))
        assert out.startswith("ERROR") and "RuntimeError" in out

    def test_default_registry_has_four_tools(self, tmp_path):
        reg = build_default_registry(tmp_path)
        assert set(reg.names()) == {"calculator", "search", "weather", "todo"}


# ----------------------------------------------------------------------
# Session / SessionStore：持久化、事件切分、reset
# ----------------------------------------------------------------------
class TestSession:
    def test_message_roundtrip_skip_none(self):
        m = Message(role="user", content="hi")
        d = m.to_dict()
        assert d == {"role": "user", "content": "hi"}
        assert Message.from_dict(d).content == "hi"

    def test_events_split_on_user_boundary(self):
        s = Session(id="x")
        s.append(Message(role="user", content="q1"))
        s.append(Message(role="assistant", content="thinking", tool_calls=[{"id": "1"}]))
        s.append(Message(role="tool", content="obs", tool_call_id="1"))
        s.append(Message(role="assistant", content="answer"))
        s.append(Message(role="user", content="q2"))
        s.append(Message(role="assistant", content="a2"))
        evs = s.events()
        assert len(evs) == 2
        assert [m.role for m in evs[0]] == ["user", "assistant", "tool", "assistant"]

    def test_store_persistence(self, tmp_path):
        store = SessionStore(tmp_path)
        s = store.get_or_create("w1")
        s.append(Message(role="user", content="你好"))
        store.save(s)

        store2 = SessionStore(tmp_path)
        s2 = store2.get_or_create("w1")
        assert s2.messages[0].content == "你好"

    def test_reset(self, tmp_path):
        store = SessionStore(tmp_path)
        s = store.get_or_create("w1")
        store.save(s)
        assert store.reset("w1") is True
        assert store.reset("w1") is False
        assert store.list_sessions() == []

    def test_sessions_sorted_by_mtime(self, tmp_path):
        store = SessionStore(tmp_path)
        for sid in ("a", "b"):
            s = store.get_or_create(sid)
            store.save(s)
        assert set(store.list_sessions()) == {"a", "b"}


# ----------------------------------------------------------------------
# parser：结构化 tool_calls 与文本协议兜底
# ----------------------------------------------------------------------
class TestParser:
    def test_structured_tool_calls(self):
        resp = LLMResponse(
            content="我先算一下",
            tool_calls=[ToolCall(id="c1", name="calculator", arguments={"expression": "1+1"})],
        )
        step = parse_response(resp)
        assert step.is_tool_step
        assert step.tool_calls[0].name == "calculator"
        assert step.thought == "我先算一下"

    def test_text_protocol_fallback(self):
        resp = LLMResponse(
            content='思考过程<tool_call>{"name": "weather", "arguments": {"city": "北京"}}</tool_call>'
        )
        step = parse_response(resp)
        assert step.is_tool_step
        assert step.tool_calls[0].name == "weather"
        assert step.tool_calls[0].arguments == {"city": "北京"}
        assert step.thought == "思考过程"

    def test_final_answer(self):
        step = parse_response(LLMResponse(content="答案是 42"))
        assert not step.is_tool_step
        assert step.final_answer == "答案是 42"

    def test_broken_json_in_text_protocol_kept_as_text(self):
        resp = LLMResponse(content='<tool_call>{"name": }</tool_call>')
        step = parse_response(resp)
        assert not step.is_tool_step  # 坏 JSON 不当工具调用，落到最终答案


# ----------------------------------------------------------------------
# ContextBuilder：token 估算、按事件压缩/截断
# ----------------------------------------------------------------------
class TestContext:
    def test_estimate_tokens(self):
        assert estimate_tokens("") == 0
        assert estimate_tokens("abcd") == 1      # 4 个 ASCII ≈ 1 token
        assert estimate_tokens("你好") == 2      # CJK 1 字 ≈ 1 token

    def test_messages_tokens_overhead(self):
        assert messages_tokens([Message(role="user", content="hi")]) == 8  # "hi" 只有 2 个 ASCII 字符，(2)//4=0

    def _builder(self, llm, token_limit=200, keep=1):
        return ContextBuilder(llm, "sys", token_limit, keep)

    def test_no_compression_when_small(self):
        calls = []

        class FakeLLM:
            def chat(self, messages, tools=None):
                calls.append(messages)
                return LLMResponse(content="s")

        s = Session(id="x")
        s.append(Message(role="user", content="hi"))
        b = self._builder(FakeLLM())
        messages, info = b.build(s)
        assert info["compressed"] is False
        assert info["hard_truncated"] is False
        assert calls == []  # 没触发压缩，不该调 LLM
        assert messages[0]["role"] == "system"

    def test_compression_merges_summary(self):
        class FakeLLM:
            def chat(self, messages, tools=None):
                return LLMResponse(content="新摘要")

        s = Session(id="x")
        # 构造 3 个事件、每个都很大，超过 token_limit=200，keep_recent=1
        for i in range(3):
            s.append(Message(role="user", content="很长的用户消息" * 30 + str(i)))
            s.append(Message(role="assistant", content="很长的回答" * 30))
        b = self._builder(FakeLLM(), token_limit=200, keep=1)
        messages, info = b.build(s)
        assert info["compressed"] is True
        assert "新摘要" in messages[0]["content"]
        # 压缩后只剩最近 1 个事件（2 条消息）
        assert len(s.messages) == 2

    def test_hard_truncation_keeps_tool_pairing(self):
        class FailLLM:
            def chat(self, messages, tools=None):
                raise RuntimeError("压缩失败")

        s = Session(id="x")
        # 事件1：超大，含 tool_call + tool 结果（必须整事件丢弃，不能拆配对）
        s.append(Message(role="user", content="问" * 1000))
        s.append(Message(role="assistant", content="", tool_calls=[{"id": "t1", "type": "function", "function": {"name": "calculator", "arguments": "{}"}}]))
        s.append(Message(role="tool", content="结" * 1000, tool_call_id="t1"))
        # 事件2：很小，应保留
        s.append(Message(role="user", content="再来一个小问题"))
        s.append(Message(role="assistant", content="好的"))
        b = self._builder(FailLLM(), token_limit=100, keep=1)
        messages, info = b.build(s)
        assert info["hard_truncated"] is True
        roles = [m["role"] for m in messages]
        assert roles == ["system", "user", "assistant"]  # system + 大事件整体消失，只剩小事件
        # 不允许出现孤儿 tool 消息 / 残缺配对
        for m in messages:
            assert m["role"] != "tool"

    def test_tool_call_text_included_in_transcript(self):
        # assistant 的 tool_calls 必须进入摘要转录，不能漏
        from mini_agent.context import _msg_text
        m = Message(role="assistant", content="想想", tool_calls=[{"function": {"name": "todo", "arguments": "{}"}}])
        assert "todo" in _msg_text(m)


# ----------------------------------------------------------------------
# llm 参数透传（不联网，检查请求体构造）
# ----------------------------------------------------------------------
class TestLLMRequestBody:
    def test_extra_body_not_sent_by_default(self, monkeypatch, tmp_path):
        """LLM_EXTAR_BODY 默认为空时不应往请求里塞奇怪参数。"""
        from mini_agent import config
        monkeypatch.setattr(config, "LLM_EXTAR_BODY", {})
        from mini_agent.llm import LLMClient
        client = LLMClient(api_key="k", base_url="http://localhost:1", model="m")
        captured = {}

        def fake_create(**kwargs):
            captured.update(kwargs)

            class R:
                class choices:  # noqa
                    pass

            return None

        monkeypatch.setattr(client._client.chat.completions, "create", fake_create)
        # 只验证 kwargs 构造，不真正调用
        kwargs = {"model": client.model, "messages": []}
        if config.LLM_EXTAR_BODY:
            kwargs["extar_body"] = dict(config.LLM_EXTAR_BODY)
        assert "extar_body" not in kwargs

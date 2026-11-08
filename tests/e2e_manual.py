"""端到端冒烟测试：真实调用 LLM API（需要 .env 里有可用 key）。

不在 pytest 默认收集范围内（文件名不带 test_ 前缀的约定之外，单独手动运行）：
    set PYTHONPATH=项目上一级目录
    python mini_agent/tests/e2e_manual.py
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

PKG_PARENT = Path(__file__).resolve().parent.parent.parent
if str(PKG_PARENT) not in sys.path:
    sys.path.insert(0, str(PKG_PARENT))

from mini_agent import config
from mini_agent.llm import LLMClient
from mini_agent.loop import Agent
from mini_agent.session import SessionStore
from mini_agent.tools import build_default_registry
from mini_agent.tracing import Tracer


def main() -> int:
    if not config.LLM_API_KEY:
        print("SKIP: 未配置 LLM_API_KEY")
        return 0

    tmp = Path(tempfile.mkdtemp(prefix="mini_agent_e2e_"))
    try:
        agent = Agent(
            LLMClient(),
            build_default_registry(tmp),
            SessionStore(tmp),
            data_dir=tmp,
            tracer=Tracer(tmp / "trace.jsonl"),
        )
        sid = "e2e"

        # ---- 场景 1：需要精确计算 => 期望模型调用 calculator ----
        r1 = agent.run(sid, "请精确计算 (3+4)*12/2 等于多少，然后告诉我。")
        used_calc = any(t["event"] == "tool_call" and t["tool"] == "calculator" for t in r1.trace)
        print(f"[场景1 计算器] stop={r1.stop_reason} turns={r1.turns} 用了calculator={used_calc}")
        print(f"  答案: {r1.final_answer}")
        assert r1.stop_reason == "final", f"stop_reason={r1.stop_reason}"
        assert used_calc, "模型没有调用 calculator 工具"
        assert "42" in (r1.final_answer or ""), f"答案不含 42: {r1.final_answer}"

        # ---- 场景 2：待办管理 => 期望调用 todo 且会话隔离 ----
        r2 = agent.run(sid, "帮我加一条待办：明天下午3点开会。")
        used_todo = any(t["event"] == "tool_call" and t["tool"] == "todo" for t in r2.trace)
        print(f"[场景2 待办] stop={r2.stop_reason} turns={r2.turns} 用了todo={used_todo}")
        assert r2.stop_reason == "final"
        assert used_todo, "模型没有调用 todo 工具"

        todo_files = list((tmp / "todos").glob("*.json"))
        assert todo_files, "待办文件没有落盘"
        data = json.loads(todo_files[0].read_text(encoding="utf-8"))
        assert any("开会" in t["text"] for t in data), f"待办内容不对: {data}"
        print(f"  落盘待办: {data}")

        # ---- 场景 3：多轮工具调用（先查后答）----
        r3 = agent.run(sid, "先帮我查一下北京今天的天气，再用 calculator 算一下如果气温升 5 度后的最高温是多少。")
        tools_used = {t["tool"] for t in r3.trace if t["event"] == "tool_call"}
        print(f"[场景3 多工具] stop={r3.stop_reason} turns={r3.turns} 用了工具={tools_used}")
        assert r3.stop_reason == "final"
        assert "weather" in tools_used, f"没有调用 weather: {tools_used}"

        # ---- 会话持久化检查 ----
        store = SessionStore(tmp)
        s = store.get_or_create(sid)
        n_user = sum(1 for m in s.messages if m.role == "user")
        print(f"[持久化] 会话共 {len(s.messages)} 条消息，其中 user {n_user} 条")
        assert n_user == 3

        print("\nE2E 冒烟测试全部通过 [PASS]")
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())

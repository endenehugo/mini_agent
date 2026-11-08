"""命令行入口：一个 session 参数 = 一个独立聊天窗口。

用法：
  python -m mini_agent.cli                 # 默认窗口 default
  python -m mini_agent.cli window1         # 窗口1（已存在则接着聊，重启不丢）
  python -m mini_agent.cli window2         # 另开一个终端 => 完全独立的会话
  python -m mini_agent.cli --list          # 列出所有会话
  python -m mini_agent.cli window1 --reset # 清空该窗口
"""
from __future__ import annotations

import argparse
import json
import sys

from . import config
from .llm import LLMClient
from .loop import Agent
from .session import SessionStore
from .tools import build_default_registry
from .tracing import Tracer


def main(argv: list[str] | None = None) -> None:
    # ---- 解析命令行参数 ----
    ap = argparse.ArgumentParser(prog="mini-agent", description="最小可用 Agent（自研 Runtime）")
    ap.add_argument("session", nargs="?", default="default", help="会话/窗口 ID（默认 default）")
    ap.add_argument("--list", action="store_true", help="列出所有会话")
    ap.add_argument("--reset", action="store_true", help="清空该会话后重新开始")
    args = ap.parse_args(argv)

    store = SessionStore(config.DATA_DIR)
    # --list / --reset 是纯本地操作，不需要 LLM，直接处理后返回
    if args.list:
        print("已有会话:", ", ".join(store.list_sessions()) or "（无）")
        return
    if args.reset:
        print("已清空" if store.reset(args.session) else "会话不存在")
        return

    # 没 key 走不到 LLM，提前失败并给出明确指引（快速失败 > 运行到一半才报错）
    if not config.LLM_API_KEY:
        sys.exit("未配置 LLM_API_KEY：请复制 .env.example 为 .env 并填入 key。")

    # ---- 组装 Agent：五个零件一次接好，之后只调 agent.run() ----
    agent = Agent(
        LLMClient(),                                  
        build_default_registry(config.DATA_DIR),     
        store,                                       
        data_dir=config.DATA_DIR,                    
        tracer=Tracer(config.TRACE_PATH),           
    )                                                

    session = store.get_or_create(args.session)
    print(
        f"mini-agent | 模型: {config.LLM_MODEL} | 会话: {args.session} | "
        f"历史 {len(session.messages)} 条消息" + ("（含早期摘要）" if session.summary else "")
    )
    print("命令: /sessions /reset /exit")

    # ---- REPL：读一行 -> 跑一轮 -> 打印结果，无限循环 ----
    while True:
        try:
            user_input = input("\n你> ").strip()
        except (EOFError, KeyboardInterrupt):  # Ctrl+D / Ctrl+C 都优雅退出
            print()
            return
        if not user_input:
            continue
        # 三个斜杠命令在本地处理，不进 Agent
        if user_input == "/exit":
            return
        if user_input == "/sessions":
            print("已有会话:", ", ".join(store.list_sessions()) or "（无）")
            continue
        if user_input == "/reset":
            store.reset(args.session)
            print("会话已清空")
            continue

        # 核心一行：把整轮任务交给 Agent 主循环
        result = agent.run(args.session, user_input)


        for r in result.trace:
            event = r["event"]
            if event == "llm_response" and r.get("thought"):
                print(f"  💭 {r['thought']}")            
            elif event == "tool_call":
                print(f"  🔧 {r['tool']}({json.dumps(r['arguments'], ensure_ascii=False)})")  
            elif event == "tool_error":
                print(f"  ⚠️ {r['tool']}: {r['error']}")
            elif event == "context_changed":
                print("  ⚙️ 上下文过长，已压缩早期对话" + ("（含硬截断）" if r.get("hard_truncated") else ""))

        # stop_reason 区分三种结束方式：正常 / 轮数耗尽 / LLM 调用失败
        if result.stop_reason == "llm_error":
            err = next((r.get("error", "") for r in result.trace if r["event"] == "llm_error"), "")
            print(f"\nAgent> [调用 LLM 失败] {err}")
        else:
            print(f"\nAgent> {result.final_answer}")
        print(f"  ({result.stop_reason}, {result.turns} 轮; trace: {config.TRACE_PATH})")


if __name__ == "__main__":
    main()

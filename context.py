from __future__ import annotations

from .llm import LLMClient
from .session import Message, Session


COMPRESS_SYSTEM_PROMPT = (
    "你是对话历史压缩器。把「已有摘要」和「待压缩对话」合并成一份新的摘要，"
    "作为后续对话的背景记忆。要求：\n"
    "1. 必须保留：用户的目标与约束、明确表达的偏好、已确认的事实与结论、"
    "待办事项及其状态、未解决的问题、关键的名词/数字/日期（保留原文）；\n"
    "2. 可以丢弃：寒暄、重复内容、工具的原始输出（只保留结论）；\n"
    "3. 中文短句分条，总长不超过 400 字，直接输出摘要内容。"
)


def estimate_tokens(text: str) -> int:
    """粗估 token：CJK 字符约 1 token/字，其他字符约 1/4 token/字符。

    只用于阈值判断，不追求精确（精确计数需要 tokenizer，这里刻意保持零依赖）。
    模型真正的分词结果与这个估计有出入，但作为「该不该压缩」的开关足够用了。
    """
    if not text:
        return 0
    # ord(ch) > 0x2E80 大致覆盖 CJK 字符区段（中文按 1 字 ≈ 1 token 计）
    cjk = sum(1 for ch in text if ord(ch) > 0x2E80)
    return cjk + (len(text) - cjk) // 4


def _msg_text(m: Message) -> str:
    """取一条消息里的纯文本（用于 token 估算与摘要转录）。

    assistant 的工具调用消息特殊：content 可能只有思考文本，还需要把
    tool_calls 里的工具名和参数拼进来，否则摘要时会漏掉「模型调了什么」。
    """
    if m.role == "assistant" and m.tool_calls:
        parts = []
        if m.content:
            parts.append(m.content)
        for tc in m.tool_calls:
            fn = tc.get("function", {})
            parts.append(f"调用工具 {fn.get('name')}({fn.get('arguments')})")
        return " ".join(parts)
    return m.content or ""


def messages_tokens(messages: list[Message]) -> int:
    """估算一组消息的 token 总量；+8 是每条消息的协议固定开销（角色等元数据）。"""
    return sum(estimate_tokens(_msg_text(m)) + 8 for m in messages)


class ContextBuilder:
    """每轮调用前执行 build()：必要先压缩，然后组装出完整 API 消息列表。"""

    def __init__(self, llm: LLMClient, system_prompt: str, token_limit: int, keep_recent_events: int):
        self.llm = llm
        self.system_prompt = system_prompt
        self.token_limit = token_limit            
        self.keep_recent_events = keep_recent_events  

    def build(self, session: Session) -> tuple[list[dict], dict]:
        """返回 (API messages, 构建信息)。构建信息含是否发生压缩、估算 token 数。"""
        info: dict = {"compressed": False, "hard_truncated": False, "est_tokens": 0}

        # ---- 第一道：软压缩（LLM 摘要） ----
        # 超阈值且还有「可压的旧事件」时，把旧事件交给 LLM 压成摘要。
        events = session.events()
        if messages_tokens(session.messages) > self.token_limit and len(events) > self.keep_recent_events:
            # 留最近 N 个事件不压，保证模型记得「刚发生的事」的原文细节
            old_events = events[: len(events) - self.keep_recent_events]
            info["compressed"] = self._compress(session, old_events)

        # ---- 第二道：硬截断兜底 ----
        # 压缩后仍超限（如单条超长工具结果、或压缩调用失败），按事件丢弃最旧的。
        # 注意丢弃的单位永远是「完整事件」，绝不会拆散 tool_call 与结果配对。
        while messages_tokens(session.messages) > self.token_limit and len(session.events()) > 1:
            drop = len(session.events()[0])
            session.messages = session.messages[drop:]
            info["hard_truncated"] = True

        # ---- 组装最终消息列表：system(带摘要) + 全部历史 ----
        messages: list[dict] = [{"role": "system", "content": self._system_with_summary(session.summary)}]
        messages.extend(m.to_dict() for m in session.messages)
        info["est_tokens"] = messages_tokens(session.messages) + estimate_tokens(messages[0]["content"])
        return messages, info

    def _system_with_summary(self, summary: str) -> str:
        """把早期摘要拼接进 system 消息尾部——摘要就是这样被模型「想起来的」。"""
        if not summary:
            return self.system_prompt
        return f"{self.system_prompt}\n\n## 早期对话摘要（更早轮次的要点）\n{summary}"

    def _compress(self, session: Session, old_events: list[list[Message]]) -> bool:
        """把旧事件压成摘要并与已有摘要合并；失败则保守跳过（不丢历史），由硬截断兜底。

        这是两级容错设计的第一级：摘要失败/为空就当无事发生，历史原样保留，
        主流程绝不中断——压缩是「锦上添花」，不能因为它让 Agent 挂掉。
        """
        old_messages = [m for ev in old_events for m in ev]
        transcript = "\n".join(f"[{m.role}] {_msg_text(m)}" for m in old_messages)
        user_prompt = (
            f"已有摘要：\n{session.summary or '（无）'}\n\n"
            f"待压缩对话：\n{transcript}\n\n请输出合并后的新摘要。"
        )
        try:
            resp = self.llm.chat(
                [
                    {"role": "system", "content": COMPRESS_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ]
            )
            new_summary = (resp.content or "").strip()
        except Exception:
            return False  
        if not new_summary:
            return False
        session.summary = new_summary
        session.messages = session.messages[len(old_messages):]  # 只按完整事件丢弃
        return True

from __future__ import annotations

import json
from dataclasses import dataclass, field

from openai import OpenAI

from . import config


@dataclass
class ToolCall:
    id : str 
    name:str
    arguments: dict

@dataclass
class LLMResponse:
    content:str | None = None
    tool_calls:list[ToolCall] = field(default_factory=list)
    finish_reason:str = ""
    usage:dict = field(default_factory=dict)

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)


class LLMClient:
    def __init__(self,api_key:str|None = None,
                 base_url:str |None = None,
                 model :str |None = None):
        self.model = model or config.LLM_MODEL
        self._client = OpenAI(
            api_key = api_key or config.LLM_API_KEY,
            base_url = base_url or config.LLM_BASE_URL,
            timeout = 60,
            max_retries = 2
        )


    def chat(self, messages:list[dict],
             tools:list[dict] | None = None) -> LLMResponse:
        """一次对话补全。tools 为 OpenAI function calling 格式的 Schema 列表。

        messages: [{"role": "system"/"user"/"assistant"/"tool", "content": ...}, ...]
        返回 LLMResponse（而不是 SDK 原始对象），让上层不依赖任何 SDK 类型。
        """
        kwargs:dict = {
            "model":self.model,
            "messages":messages,
            "temperature": 0.3,
        }
        if tools :
            kwargs["tools"] = tools
        if config.LLM_EXTAR_BODY:
            kwargs["extar_body"] = dict(config.LLM_EXTAR_BODY)

        resp = self._client.chat.completions.create(**kwargs)

        choice = resp.choices[0]
        msg = choice.message
        calls:list[ToolCall] = []
        for tc in getattr(msg,"tool_calls", None) or []:
            args: dict | None
            try:
                parsed = json.loads(tc.function.arguments) if tc.function.arguments else {}
                args = parsed if isinstance(parsed,dict) else {"_raw":parsed}
            except(json.JSONDecodeError,TypeError):
                args = None
            calls.append(ToolCall(
                id = tc.id,
                name = tc.function.name,
                arguments = args
            ))

        usage = getattr(resp,"usage",None)

        return LLMResponse(
            content=msg.content,
            tool_calls=calls,
            finish_reason=choice.finish_reason or "",
            usage={
                "prompt_tokens":getattr(usage,"prompt_tokens",0) or 0,
                "completion_tokens":getattr(usage,"completion_tokens",0) or 0,
            }if usage else {},
        )
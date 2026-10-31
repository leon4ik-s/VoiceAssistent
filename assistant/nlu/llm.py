"""Клиент языковой модели через OpenAI-совместимый API (Ollama локально или OpenAI).

Локальные модели хуже держат function calling, поэтому здесь есть страховки:
- мягкий разбор аргументов (код в ```, одинарные кавычки, лишние запятые);
- вызов инструмента, который модель написала текстом (<tool_call>{...}</tool_call> или голый JSON);
- JSON-режим без tools, если сервер или модель не поддерживает tools (LLM_TOOLS_MODE=json или автоматически).
"""
from __future__ import annotations

import ast
import json
import logging
import re
import uuid
from dataclasses import dataclass, field
from typing import Any, Protocol

log = logging.getLogger(__name__)


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict
    error: str | None = None  # аргументы не удалось разобрать


@dataclass
class LLMResponse:
    content: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)

    def as_message(self) -> dict:
        """Сообщение ассистента в формате Chat Completions (для продолжения диалога с инструментами)."""
        msg: dict[str, Any] = {"role": "assistant", "content": self.content or ""}
        if self.tool_calls:
            msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.name, "arguments": json.dumps(tc.arguments, ensure_ascii=False)},
                }
                for tc in self.tool_calls
            ]
        return msg


class LLMClient(Protocol):
    def chat(self, messages: list[dict], tools: list[dict]) -> LLMResponse: ...


# ---------------------------------------------------------------------- разбор ответов
def parse_json_lenient(raw: Any) -> Any:
    """Разобрать JSON, который модель могла написать небрежно. ValueError, если не получилось."""
    if isinstance(raw, (dict, list)):
        return raw
    text = str(raw or "").strip()
    if not text:
        return {}
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.S).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{.*\}", text, flags=re.S)  # JSON внутри текста
    candidate = m.group(0) if m else text
    candidate = re.sub(r",\s*([}\]])", r"\1", candidate)
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass
    try:  # одинарные кавычки, True/False/None
        lit = candidate.replace("true", "True").replace("false", "False").replace("null", "None")
        value = ast.literal_eval(lit)
        if isinstance(value, (dict, list)):
            return value
    except (ValueError, SyntaxError):
        pass
    raise ValueError(f"не JSON: {text[:200]}")


def _new_id() -> str:
    return "call_" + uuid.uuid4().hex[:12]


def make_tool_call(name: str, raw_args: Any, call_id: str | None = None) -> ToolCall:
    try:
        args = parse_json_lenient(raw_args)
        if not isinstance(args, dict):
            raise ValueError("аргументы должны быть JSON-объектом")
        return ToolCall(id=call_id or _new_id(), name=name, arguments=args)
    except ValueError as exc:
        return ToolCall(id=call_id or _new_id(), name=name, arguments={}, error=str(exc))


TOOL_TAG = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.S)


def tool_calls_from_text(content: str | None, tool_names: set[str]) -> list[ToolCall]:
    """Достать вызовы инструментов, которые модель написала текстом вместо tool_calls."""
    if not content or ("{" not in content):
        return []
    chunks = TOOL_TAG.findall(content) or [content]
    calls = []
    for chunk in chunks:
        try:
            obj = parse_json_lenient(chunk)
        except ValueError:
            continue
        for o in obj if isinstance(obj, list) else [obj]:
            if not isinstance(o, dict):
                continue
            name = o.get("name") or o.get("tool") or (o.get("function") or {}).get("name")
            args = o.get("arguments", o.get("parameters", (o.get("function") or {}).get("arguments", {})))
            if name in tool_names:
                calls.append(make_tool_call(name, args))
    return calls


def _tools_unsupported(exc: Exception) -> bool:
    status = getattr(exc, "status_code", None)
    text = str(exc).lower()
    return status in (400, 404, 422, 500) and "tool" in text and ("support" in text or "not" in text)


# ---------------------------------------------------------------------- JSON-режим
JSON_MODE_INSTRUCTIONS = """
Инструменты (вызывай по одному за ответ):
{tools}
Отвечай СТРОГО одним JSON-объектом без пояснений:
- вызвать инструмент: {{"tool": "имя", "arguments": {{...}}}}
- ответить пользователю: {{"reply": "текст"}}"""


def _tools_brief(tools: list[dict]) -> str:
    lines = []
    for t in tools:
        f = t["function"]
        params = ", ".join(
            f"{k}{'*' if k in f['parameters'].get('required', []) else ''}"
            for k in f["parameters"]["properties"]
        )
        lines.append(f"- {f['name']}({params}): {f['description']}")
    return "\n".join(lines) + "\n(* - обязательный параметр)"


def to_json_mode_messages(messages: list[dict], tools: list[dict]) -> list[dict]:
    """Переписать диалог с tool-сообщениями в обычный чат для JSON-режима."""
    out = []
    for m in messages:
        role = m["role"]
        if role == "system" and not out:
            out.append({"role": "system", "content": m["content"] + JSON_MODE_INSTRUCTIONS.format(tools=_tools_brief(tools))})
        elif role == "assistant" and m.get("tool_calls"):
            calls = [{"tool": c["function"]["name"], "arguments": json.loads(c["function"]["arguments"])}
                     for c in m["tool_calls"]]
            out.append({"role": "assistant", "content": json.dumps(calls[0] if len(calls) == 1 else calls, ensure_ascii=False)})
        elif role == "tool":
            out.append({"role": "user", "content": "Результат инструмента: " + m["content"]})
        elif role == "assistant":
            out.append({"role": "assistant", "content": json.dumps({"reply": m.get("content") or ""}, ensure_ascii=False)})
        else:
            out.append({"role": role, "content": m.get("content") or ""})
    return out


def parse_json_mode_reply(content: str | None, tool_names: set[str]) -> LLMResponse:
    try:
        obj = parse_json_lenient(content or "")
    except ValueError:
        return LLMResponse(content=(content or "").strip())  # модель ответила текстом - так и отдаем
    objs = obj if isinstance(obj, list) else [obj]
    calls = []
    for o in objs:
        if isinstance(o, dict) and (o.get("tool") or o.get("name")) in tool_names:
            calls.append(make_tool_call(o.get("tool") or o.get("name"), o.get("arguments") or {}))
    if calls:
        return LLMResponse(tool_calls=calls)
    if isinstance(obj, dict):
        return LLMResponse(content=str(obj.get("reply") or obj.get("content") or obj.get("text") or "").strip())
    return LLMResponse(content=(content or "").strip())


# ---------------------------------------------------------------------- клиент
class OpenAIChat:
    """Chat Completions с tools. Работает с OpenAI и с Ollama (base_url=http://localhost:11434/v1)."""

    def __init__(self, client, model: str, temperature: float | None = 0.0, tools_mode: str = "auto"):
        self.client = client
        self.model = model
        self.temperature = temperature
        self.tools_mode = tools_mode if tools_mode in ("auto", "tools", "json") else "auto"

    def _kwargs(self) -> dict:
        kw: dict = {"model": self.model}
        # у reasoning-моделей OpenAI (o*, gpt-5) температура фиксирована
        if self.temperature is not None and not self.model.startswith(("o1", "o3", "o4", "gpt-5")):
            kw["temperature"] = self.temperature
        return kw

    def chat(self, messages: list[dict], tools: list[dict]) -> LLMResponse:
        names = {t["function"]["name"] for t in tools}
        if self.tools_mode != "json":
            try:
                return self._chat_tools(messages, tools, names)
            except Exception as exc:
                if self.tools_mode == "tools" or not _tools_unsupported(exc):
                    raise
                log.warning("Модель %s не поддерживает tools, перехожу в JSON-режим: %s", self.model, exc)
                self.tools_mode = "json"
        return self._chat_json(messages, tools, names)

    def _chat_tools(self, messages, tools, names) -> LLMResponse:
        resp = self.client.chat.completions.create(messages=messages, tools=tools, tool_choice="auto", **self._kwargs())
        msg = resp.choices[0].message
        calls = [make_tool_call(tc.function.name, tc.function.arguments, tc.id) for tc in (msg.tool_calls or [])]
        if not calls:
            calls = tool_calls_from_text(msg.content, names)
            if calls:
                return LLMResponse(tool_calls=calls)
        return LLMResponse(content=msg.content, tool_calls=calls)

    def _chat_json(self, messages, tools, names) -> LLMResponse:
        kw = self._kwargs()
        jm = to_json_mode_messages(messages, tools)
        try:
            resp = self.client.chat.completions.create(messages=jm, response_format={"type": "json_object"}, **kw)
        except Exception as exc:  # сервер без response_format - пробуем без него
            if getattr(exc, "status_code", None) not in (400, 422):
                raise
            resp = self.client.chat.completions.create(messages=jm, **kw)
        return parse_json_mode_reply(resp.choices[0].message.content, names)

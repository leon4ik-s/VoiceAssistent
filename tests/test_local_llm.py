"""Устойчивость к локальным моделям: битый JSON, вызовы текстом, JSON-режим без tools, валидация, провайдеры."""
import json
import threading
from datetime import date
from http.server import BaseHTTPRequestHandler, HTTPServer
from types import SimpleNamespace as NS

import pytest

from assistant.config import ConfigError, Settings
from assistant.nlu.agent import Assistant
from assistant.nlu.llm import OpenAIChat, parse_json_lenient, tool_calls_from_text
from assistant.nlu.tools import TOOLS, ArgError, validate_args
from assistant.providers import build_providers, ollama_health
from assistant.storage.embeddings import EmbeddingIndex

from conftest import FakeEmbedder, ScriptedLLM

NAMES = {t["function"]["name"] for t in TOOLS}
TODAY = date(2026, 10, 30)


# ---------------------------------------------------------------- разбор ответов модели
@pytest.mark.parametrize("raw", [
    '{"title": "Встреча", "time": "15:00"}',
    '```json\n{"title": "Встреча", "time": "15:00"}\n```',
    "{'title': 'Встреча', 'time': '15:00'}",
    '{"title": "Встреча", "time": "15:00",}',
    'Вызываю: {"title": "Встреча", "time": "15:00"}',
])
def test_lenient_json(raw):
    assert parse_json_lenient(raw) == {"title": "Встреча", "time": "15:00"}


def test_lenient_json_rejects_garbage():
    with pytest.raises(ValueError):
        parse_json_lenient("создай встречу завтра")


def test_tool_call_written_as_text():
    text = '<tool_call>\n{"name": "create_item", "arguments": {"kind": "task", "title": "Отчет"}}\n</tool_call>'
    calls = tool_calls_from_text(text, NAMES)
    assert [(c.name, c.arguments["title"]) for c in calls] == [("create_item", "Отчет")]
    bare = '{"name": "search_items", "arguments": {"query": "договор"}}'
    assert tool_calls_from_text(bare, NAMES)[0].name == "search_items"
    assert tool_calls_from_text("Записал встречу на завтра.", NAMES) == []
    assert tool_calls_from_text('{"name": "rm_rf", "arguments": {}}', NAMES) == []


# ---------------------------------------------------------------- клиент с фейковым API
class FakeAPI:
    """Имитация openai.OpenAI(...).chat.completions с заранее заданными ответами."""

    def __init__(self, replies, reject_tools=False):
        self.replies = list(replies)
        self.reject_tools = reject_tools
        self.calls = []
        self.chat = NS(completions=NS(create=self.create))

    def create(self, **kw):
        self.calls.append(kw)
        if self.reject_tools and "tools" in kw:
            err = Exception("registry.ollama.ai/library/tiny does not support tools")
            err.status_code = 400
            raise err
        content, tool_calls = self.replies.pop(0)
        tcs = [NS(id=f"c{i}", function=NS(name=n, arguments=a)) for i, (n, a) in enumerate(tool_calls or [])]
        return NS(choices=[NS(message=NS(content=content, tool_calls=tcs or None))])


def test_native_tools_bad_json_and_temperature():
    api = FakeAPI([(None, [("create_item", '{"kind": "task", "title": ')])])
    resp = OpenAIChat(api, "qwen2.5:7b", temperature=0.0).chat([{"role": "user", "content": "x"}], TOOLS)
    assert resp.tool_calls[0].error and api.calls[0]["temperature"] == 0.0
    assert api.calls[0]["tool_choice"] == "auto"


def test_fallback_to_json_mode_when_tools_unsupported():
    api = FakeAPI([
        ('{"tool": "create_item", "arguments": {"kind": "task", "title": "Отчет", "date": "2026-10-31"}}', None),
        ('{"reply": "Записал."}', None),
    ], reject_tools=True)
    llm = OpenAIChat(api, "tiny", tools_mode="auto")
    msgs = [{"role": "system", "content": "sys"}, {"role": "user", "content": "Задача отчет на завтра"}]
    r1 = llm.chat(msgs, TOOLS)
    assert llm.tools_mode == "json" and r1.tool_calls[0].name == "create_item"
    json_call = api.calls[1]
    assert "tools" not in json_call and json_call["response_format"] == {"type": "json_object"}
    assert "create_item(" in json_call["messages"][0]["content"]
    # продолжение диалога: tool-сообщения переписаны в обычный чат
    msgs += [r1.as_message(), {"role": "tool", "tool_call_id": "c0", "content": '{"status": "ok"}'}]
    r2 = llm.chat(msgs, TOOLS)
    assert r2.content == "Записал." and not r2.tool_calls
    assert all(m["role"] in ("system", "user", "assistant") for m in api.calls[2]["messages"])


# ---------------------------------------------------------------- строгая валидация аргументов
def test_validate_coerces_and_normalizes():
    a = validate_args("create_item", {
        "kind": "Meeting", "title": " Встреча ", "date": "завтра", "time": "15", "duration_min": "30",
        "force": "false", "junk": 1, "recurrence": "weekly",
    }, TODAY)
    assert a == {"kind": "meeting", "title": "Встреча", "date": "2026-10-31", "time": "15:00",
                 "duration_min": 30, "force": False, "recurrence": {"freq": "weekly"}}
    assert validate_args("list_schedule", {"date_from": "02.11", "date_to": "8.11.2026"}, TODAY) == \
        {"date_from": "2026-11-02", "date_to": "2026-11-08"}


@pytest.mark.parametrize("name,args", [
    ("create_item", {"kind": "event", "title": "X"}),
    ("create_item", {"kind": "task"}),
    ("update_item", {"item_id": "первая"}),
    ("search_items", {}),
    ("drop_database", {}),
])
def test_validate_rejects(name, args):
    with pytest.raises(ArgError):
        validate_args(name, args, TODAY)


# ---------------------------------------------------------------- агент с капризной моделью
def test_agent_recovers_from_broken_arguments(ctx, user):
    llm = ScriptedLLM([
        lambda m: _broken_call(),
        [("create_item", {"kind": "meeting", "title": "Встреча с Иваном", "date": "завтра", "time": "15:00"})],
        "Записал встречу на завтра в 15:00.",
    ])
    a = Assistant(ctx.repo, ctx.scheduler, lambda: llm)
    res = a.handle(user, "Встреча с Иваном завтра в 15")
    assert res.changed and [x["status"] for x in res.actions] == ["error", "ok"]
    assert "JSON" in json.loads(llm.requests[1][-1]["content"])["message"]
    assert ctx.scheduler.schedule(user, "2026-10-31")[0]["time"] == "15:00"


def _broken_call():
    from assistant.nlu.llm import LLMResponse, make_tool_call

    return LLMResponse(tool_calls=[make_tool_call("create_item", '{"kind": "meeting", "title": ')])


def test_agent_dialog_in_json_mode(ctx, user):
    """Полный сценарий через JSON-режим: создать, уточнение, «перенеси ее на завтра», история не ломается."""
    api = FakeAPI([
        ('{"tool": "create_item", "arguments": {"kind": "meeting", "title": "Созвон с Марией", "date": "2026-10-30"}}', None),
        ('{"reply": "Во сколько созвон?"}', None),
        ('{"tool": "create_item", "arguments": {"kind": "meeting", "title": "Созвон с Марией", "date": "2026-10-30", "time": "16:00"}}', None),
        ('{"reply": "Записал на 16:00."}', None),
        ('{"tool": "update_item", "arguments": {"date": "2026-10-31"}}', None),
        ('Перенес на завтра.', None),
    ], reject_tools=True)
    llm = OpenAIChat(api, "tiny")
    a = Assistant(ctx.repo, ctx.scheduler, lambda: llm)
    assert a.handle(user, "Созвон с Марией сегодня").reply == "Во сколько созвон?"
    assert ctx.repo.count_items(user) == 0
    a.handle(user, "В 16")
    assert a.handle(user, "Перенеси ее на завтра").reply == "Перенес на завтра."
    item = ctx.repo.list_items(user)[0]
    assert (item.date, item.time) == ("2026-10-31", "16:00")
    roles = [m["role"] for m in ctx.repo.recent_messages(user, 20)]
    assert roles.count("user") == 3 and roles.count("assistant") == 3


# ---------------------------------------------------------------- провайдеры и проверка Ollama
def test_local_providers_without_whisper(monkeypatch, tmp_path):
    import importlib.util as iu

    real = iu.find_spec
    monkeypatch.setattr(iu, "find_spec", lambda name, *a: None if name == "faster_whisper" else real(name, *a))
    p = build_providers(Settings(provider="local", db_path=tmp_path / "x.db"))
    assert p.stt_server is False and p.tts_server is False and p.embedder.model == "bge-m3"
    with pytest.raises(ConfigError, match="requirements-whisper"):
        p.stt()
    with pytest.raises(ConfigError):
        p.tts()
    assert p.llm().model == "qwen2.5:7b" and p.llm().temperature == 0.0


def test_openai_provider_requires_key(tmp_path):
    p = build_providers(Settings(provider="openai", db_path=tmp_path / "x.db"))
    with pytest.raises(ConfigError, match="OPENAI_API_KEY"):
        p.llm()
    assert p.health()["ok"] is False


class _Tags(BaseHTTPRequestHandler):
    def do_GET(self):
        body = json.dumps({"models": [{"name": "qwen2.5:7b"}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


def test_ollama_health_reports_missing_models():
    srv = HTTPServer(("127.0.0.1", 0), _Tags)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        s = Settings(llm_base_url=f"http://127.0.0.1:{srv.server_port}/v1")
        h = ollama_health(s)
        assert h["ok"] is False and "ollama pull bge-m3" in h["message"] and "qwen" not in h["message"]
        s.embed_model = "qwen2.5:7b"
        assert ollama_health(s)["ok"] is True
    finally:
        srv.shutdown()


def test_ollama_health_when_not_running():
    h = ollama_health(Settings(llm_base_url="http://127.0.0.1:9/v1"), timeout=0.5)
    assert h["ok"] is False and "ollama serve" in h["message"]


def test_embedding_cache_is_per_model(ctx, user):
    ctx.scheduler.create(user, "task", "Отчет по бюджету", "2026-11-01")
    e1, e2 = FakeEmbedder(), FakeEmbedder()
    e2.model = "other-model"
    EmbeddingIndex(ctx.repo, e1).vectors_for(["Отчет по бюджету"])
    EmbeddingIndex(ctx.repo, e1).vectors_for(["Отчет по бюджету"])
    EmbeddingIndex(ctx.repo, e2).vectors_for(["Отчет по бюджету"])
    assert len(e1.calls) == 1 and len(e2.calls) == 1   # смена модели - пересчет, та же модель - кэш

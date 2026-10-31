"""Общие фикстуры: фейковые LLM, эмбеддинги, STT/TTS и фиксированное «сейчас»."""
from __future__ import annotations

import hashlib
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from assistant.config import ConfigError, Settings  # noqa: E402
from assistant.nlu.llm import LLMResponse, ToolCall  # noqa: E402
from assistant.providers import Providers, build_context  # noqa: E402
from assistant.services.search import tokens  # noqa: E402

CSV_PATH = ROOT / "data" / "voice_assistant_events_1000_77e927b5d2.csv"
NOW = datetime(2026, 10, 30, 10, 0)  # пятница


class FakeEmbedder:
    """Детерминированные «смысловые» векторы: мешок основ слов (4 буквы), захешированный в 256 измерений."""

    model = "fake-embed"

    def __init__(self):
        self.calls: list[list[str]] = []

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        out = []
        for t in texts:
            v = np.zeros(256, dtype=np.float32)
            for s in {w[:4] for w in tokens(t)}:
                v[int(hashlib.md5(s.encode()).hexdigest(), 16) % 256] += 1.0
            out.append(v.tolist())
        return out


class ScriptedLLM:
    """Возвращает заранее заданные ответы. Шаг: строка (финальный ответ) или список (имя, аргументы)."""

    def __init__(self, steps):
        self.steps = list(steps)
        self.requests: list[list[dict]] = []

    def chat(self, messages, tools):
        self.requests.append([dict(m) for m in messages])
        step = self.steps.pop(0)
        if callable(step):
            step = step(messages)
        if isinstance(step, LLMResponse):
            return step
        if isinstance(step, str):
            return LLMResponse(content=step)
        return LLMResponse(tool_calls=[ToolCall(id=f"call_{i}", name=n, arguments=a) for i, (n, a) in enumerate(step)])


class FakeSTT:
    def __init__(self, text="напомни завтра в 9 позвонить Ивану"):
        self.text = text

    def transcribe(self, audio, filename="audio.webm", content_type=None):
        return self.text


class FakeTTS:
    def synthesize(self, text):
        return b"ID3fake-mp3"


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(provider="local", db_path=tmp_path / "test.db", history_csv=CSV_PATH,
                    import_csv_on_start=False, default_user="Основной")


@pytest.fixture
def llm_holder():
    return {"llm": ScriptedLLM([])}


@pytest.fixture
def embedder():
    return FakeEmbedder()


@pytest.fixture
def providers(llm_holder, embedder) -> Providers:
    return Providers(llm=lambda: llm_holder["llm"], stt=lambda: FakeSTT(), tts=lambda: FakeTTS(), embedder=embedder)


OLLAMA_DOWN = "Ollama не отвечает по адресу http://localhost:11434. Установите Ollama с ollama.com и запустите"


@pytest.fixture
def no_key_providers() -> Providers:
    """Языковая модель недоступна (Ollama не запущен), серверного STT нет, озвучка в браузере."""
    def missing():
        raise ConfigError(OLLAMA_DOWN)

    return Providers(llm=missing, stt=missing, tts=missing, embedder=None,
                     health=lambda: {"ok": False, "message": OLLAMA_DOWN}, stt_server=False, tts_server=False)


@pytest.fixture
def ctx(settings, providers):
    c = build_context(settings, providers, now_fn=lambda: NOW)
    yield c
    c.db.close()


@pytest.fixture
def user(ctx):
    return ctx.repo.get_or_create_user("Анна")["id"]


@pytest.fixture
def other_user(ctx):
    return ctx.repo.get_or_create_user("Борис")["id"]

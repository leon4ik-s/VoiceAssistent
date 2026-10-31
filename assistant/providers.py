"""Сборка внешних сервисов и всего приложения из настроек.

PROVIDER=yandex (по умолчанию): YandexGPT через OpenAI-совместимый API AI Studio, эмбеддинги
text-search-doc/query, распознавание и озвучка - SpeechKit.
PROVIDER=local: Ollama + faster-whisper/браузер, бесплатно. PROVIDER=openai: все через OpenAI.
Все сервисы скрыты за простыми интерфейсами (LLMClient, SpeechToText, TextToSpeech, Embedder),
поэтому в тестах они заменяются фейками.
"""
from __future__ import annotations

import importlib.util
import json
import logging
import re
import urllib.request
from dataclasses import dataclass, field
from typing import Callable

from .config import ConfigError, Settings
from .nlu.agent import Assistant
from .nlu.llm import LLMClient, OpenAIChat
from .services.scheduler import Scheduler
from .services.search import SemanticSearch
from .storage.db import Database
from .storage.embeddings import Embedder, EmbeddingIndex
from .storage.repository import Repository
from .voice.speech import FasterWhisperSTT, OpenAISpeechToText, OpenAITextToSpeech, SpeechToText, TextToSpeech
from .voice.yandex_speech import MAX_STT_SECONDS, SpeechKitSTT, SpeechKitTTS
from .yandex_cloud import YandexAPIError, YandexEmbedder, yandex_error_message

log = logging.getLogger(__name__)

NO_KEY_MESSAGE = (
    "Выбран PROVIDER=openai, но не задан OPENAI_API_KEY. Впишите ключ в .env или верните PROVIDER=local "
    "(бесплатно, через Ollama) и перезапустите сервер."
)
NO_WHISPER_MESSAGE = (
    "Серверное распознавание речи не установлено. Выполните: pip install -r requirements-whisper.txt "
    "и перезапустите сервер. Пока можно распознавать речь в браузере (Chrome)."
)
BROWSER_STT_MESSAGE = "Серверное распознавание выключено (STT_ENGINE=browser): используется распознавание в браузере."
BROWSER_TTS_MESSAGE = "Серверная озвучка выключена: ответы озвучивает браузер."
NO_YANDEX_MESSAGE = (
    "Не заданы ключи Yandex Cloud. Впишите YANDEX_API_KEY и YANDEX_FOLDER_ID в файл .env "
    "(как их получить - в README) и перезапустите сервер. Расписание работает и без ключей."
)


class OpenAIEmbedder:
    """Эмбеддинги через /v1/embeddings (OpenAI или Ollama)."""

    def __init__(self, client, model: str):
        self.client = client
        self.model = model

    def embed(self, texts: list[str]) -> list[list[float]]:
        resp = self.client.embeddings.create(model=self.model, input=texts)
        return [d.embedding for d in sorted(resp.data, key=lambda d: d.index)]


@dataclass
class Providers:
    """Фабрики внешних сервисов. Недоступный сервис бросает ConfigError с понятным текстом."""

    llm: Callable[[], LLMClient]
    stt: Callable[[], SpeechToText]
    tts: Callable[[], TextToSpeech]
    embedder: Embedder | None  # None - работает только словесная часть поиска
    health: Callable[[], dict] = field(default=lambda: {"ok": True, "message": None})
    stt_server: bool = True    # есть ли серверное распознавание
    tts_server: bool = True    # есть ли серверная озвучка
    stt_max_seconds: int = 120  # ограничение длины фразы (SpeechKit - 30 секунд)


def _raiser(message: str):
    def fail():
        raise ConfigError(message)

    return fail


# ---------------------------------------------------------------------- Ollama
def ollama_root(base_url: str) -> str:
    return re.sub(r"/v1/?$", "", base_url.rstrip("/"))


def model_available(name: str, installed: list[str]) -> bool:
    if name in installed:
        return True
    if ":" not in name:
        return f"{name}:latest" in installed
    return False


def ollama_health(settings: Settings, timeout: float = 2.0) -> dict:
    """Проверить, что Ollama запущен и нужные модели скачаны. Возвращает {ok, message, models}."""
    root = ollama_root(settings.llm_base_url)
    try:
        with urllib.request.urlopen(f"{root}/api/tags", timeout=timeout) as r:
            installed = [m.get("name", "") for m in json.loads(r.read().decode("utf-8")).get("models", [])]
    except Exception:
        try:  # другой OpenAI-совместимый сервер (LM Studio и т.п.)
            with urllib.request.urlopen(f"{settings.llm_base_url.rstrip('/')}/models", timeout=timeout) as r:
                installed = [m.get("id", "") for m in json.loads(r.read().decode("utf-8")).get("data", [])]
        except Exception:
            return {
                "ok": False,
                "models": [],
                "message": f"Ollama не отвечает по адресу {root}. Установите Ollama с ollama.com и запустите "
                           f"приложение (или команду «ollama serve»), затем обновите страницу.",
            }
    need = [settings.llm_model] + ([settings.embed_model] if settings.embed_model else [])
    missing = [m for m in dict.fromkeys(need) if not model_available(m, installed)]
    if missing:
        cmds = "; ".join(f"ollama pull {m}" for m in missing)
        return {"ok": False, "models": installed,
                "message": f"Не скачаны модели: {', '.join(missing)}. Выполните в терминале: {cmds}"}
    return {"ok": True, "models": installed, "message": None}


def describe_error(exc: Exception, settings: Settings | None = None) -> tuple[int, str]:
    """Перевести исключение внешнего сервиса в понятное сообщение для интерфейса."""
    if isinstance(exc, ConfigError):
        return 503, str(exc)
    if isinstance(exc, YandexAPIError):
        return yandex_error_message(exc.status_code, exc.text, exc.service)
    local = settings is not None and settings.provider == "local"
    yandex = settings is not None and settings.is_yandex
    try:
        import openai
    except ImportError:  # pragma: no cover
        openai = None
    if openai is not None and isinstance(exc, openai.OpenAIError):
        text = getattr(exc, "message", None) or str(exc)
        if yandex:
            if isinstance(exc, (openai.APIConnectionError, openai.APITimeoutError)):
                return yandex_error_message(None, text, "YandexGPT")
            return yandex_error_message(getattr(exc, "status_code", None) or 500, text, "YandexGPT")
        if local:
            if isinstance(exc, (openai.APIConnectionError, openai.APITimeoutError)):
                return 503, (f"Нет связи с Ollama ({ollama_root(settings.llm_base_url)}). Запустите приложение "
                             f"Ollama или команду «ollama serve» и повторите.")
            if isinstance(exc, openai.NotFoundError) or "not found" in text.lower():
                m = re.search(r"model ['\"]?([\w.:\-/]+)['\"]? not found", text)
                model = m.group(1) if m else settings.llm_model
                return 503, f"Модель {model} не скачана в Ollama. Выполните в терминале: ollama pull {model}"
            return 502, f"Ошибка локальной модели: {text}"
        if isinstance(exc, openai.AuthenticationError):
            return 502, "OpenAI отклонил ключ. Проверьте OPENAI_API_KEY в .env."
        if isinstance(exc, openai.PermissionDeniedError):
            return 502, ("OpenAI запретил доступ (403). Возможно, регион не поддерживается: "
                         "укажите OPENAI_BASE_URL прокси-сервера или верните PROVIDER=local.")
        if isinstance(exc, openai.RateLimitError):
            return 429, "Превышен лимит запросов или закончились средства на балансе OpenAI."
        if isinstance(exc, (openai.APIConnectionError, openai.APITimeoutError)):
            return 502, "Нет связи с OpenAI. Проверьте интернет, VPN или OPENAI_BASE_URL."
        return 502, f"Ошибка OpenAI: {text}"
    if isinstance(exc, (ImportError, OSError, RuntimeError)) and "whisper" in repr(exc).lower():
        return 503, f"Ошибка распознавания речи: {exc}"
    return 500, f"Внутренняя ошибка: {exc}"


# ---------------------------------------------------------------------- сборка
def build_providers(settings: Settings) -> Providers:
    from openai import OpenAI

    openai_client = None
    if settings.openai_configured:
        openai_client = OpenAI(api_key=settings.openai_api_key, base_url=settings.openai_base_url,
                               timeout=60, max_retries=2)

    yandex_ok = settings.yandex_configured
    ykey, ytoken, yfolder = settings.yandex_api_key, settings.yandex_iam_token, settings.yandex_folder_id

    # LLM и эмбеддинги
    if settings.is_yandex:
        if not yandex_ok:
            llm, embedder = _raiser(NO_YANDEX_MESSAGE), None
            health = lambda: {"ok": False, "message": NO_YANDEX_MESSAGE}  # noqa: E731
        else:
            # OpenAI-совместимый API AI Studio: ключ уходит как Bearer, каталог - в заголовке OpenAI-Project
            yc = OpenAI(api_key=ykey or ytoken, base_url=settings.yandex_llm_base_url, project=yfolder,
                        timeout=60, max_retries=1)
            chat = OpenAIChat(yc, settings.yandex_gpt_uri, temperature=settings.yandex_temperature,
                              tools_mode=settings.llm_tools_mode)
            llm = lambda: chat  # noqa: E731
            embedder = YandexEmbedder(yc, yfolder, settings.yandex_embed_doc_model,
                                      settings.yandex_embed_query_model, settings.yandex_embed_rps)
            health = lambda: {"ok": True, "message": None}  # noqa: E731
    elif settings.is_openai:
        if openai_client is None:
            llm, embedder = _raiser(NO_KEY_MESSAGE), None
        else:
            chat = OpenAIChat(openai_client, settings.openai_chat_model, temperature=0.2, tools_mode="tools")
            llm, embedder = (lambda: chat), OpenAIEmbedder(openai_client, settings.openai_embed_model)
        health = (lambda: {"ok": True, "message": None}) if openai_client else (lambda: {"ok": False, "message": NO_KEY_MESSAGE})
    else:
        local = OpenAI(api_key=settings.llm_api_key or "ollama", base_url=settings.llm_base_url, timeout=180, max_retries=0)
        chat = OpenAIChat(local, settings.llm_model, temperature=settings.llm_temperature,
                          tools_mode=settings.llm_tools_mode)
        llm = lambda: chat  # noqa: E731
        emb_client = local if not settings.embed_base_url else OpenAI(
            api_key=settings.llm_api_key or "ollama", base_url=settings.embed_base_url, timeout=180, max_retries=0)
        embedder = OpenAIEmbedder(emb_client, settings.embed_model) if settings.embed_model else None
        health = lambda: ollama_health(settings)  # noqa: E731

    # Распознавание речи
    stt_server, stt_max = True, 120
    engine = settings.resolved_stt
    if engine == "yandex":
        stt_max = MAX_STT_SECONDS - 1
        if not yandex_ok:
            stt, stt_server = _raiser(NO_YANDEX_MESSAGE), False
        else:
            skstt = SpeechKitSTT(ykey, yfolder, ytoken, settings.stt_language, settings.yandex_stt_url)
            stt = lambda: skstt  # noqa: E731
    elif engine == "local":
        if importlib.util.find_spec("faster_whisper") is None:
            stt, stt_server = _raiser(NO_WHISPER_MESSAGE), False
        else:
            whisper = FasterWhisperSTT(settings.whisper_model, settings.whisper_device,
                                       settings.whisper_compute_type, settings.stt_language)
            stt = lambda: whisper  # noqa: E731
    elif engine == "openai":
        if openai_client is None:
            stt, stt_server = _raiser("Для STT_ENGINE=openai нужен OPENAI_API_KEY в .env."), False
        else:
            ostt = OpenAISpeechToText(openai_client, settings.openai_stt_model, settings.stt_language)
            stt = lambda: ostt  # noqa: E731
    else:
        stt, stt_server = _raiser(BROWSER_STT_MESSAGE), False

    # Озвучка
    tts_server = False
    if settings.resolved_tts == "yandex" and yandex_ok:
        sktts = SpeechKitTTS(ykey, yfolder, ytoken, settings.yandex_tts_voice, settings.yandex_tts_speed,
                             settings.yandex_tts_url)
        tts, tts_server = (lambda: sktts), True
    elif settings.resolved_tts == "yandex":
        tts = _raiser("Для озвучки SpeechKit нужны YANDEX_API_KEY и YANDEX_FOLDER_ID. Сейчас ответы озвучивает браузер.")
    elif settings.resolved_tts == "openai" and openai_client is not None:
        otts = OpenAITextToSpeech(openai_client, settings.openai_tts_model, settings.openai_tts_voice)
        tts, tts_server = (lambda: otts), True
    elif settings.resolved_tts == "openai":
        tts = _raiser("Для серверной озвучки нужен OPENAI_API_KEY. Сейчас ответы озвучивает браузер.")
    else:
        tts = _raiser(BROWSER_TTS_MESSAGE)

    return Providers(llm=llm, stt=stt, tts=tts, embedder=embedder, health=health,
                     stt_server=stt_server, tts_server=tts_server, stt_max_seconds=stt_max)


@dataclass
class AppContext:
    settings: Settings
    db: Database
    repo: Repository
    search: SemanticSearch
    scheduler: Scheduler
    assistant: Assistant
    providers: Providers


def build_context(settings: Settings, providers: Providers | None = None, now_fn=None) -> AppContext:
    from datetime import datetime
    from zoneinfo import ZoneInfo

    providers = providers or build_providers(settings)
    db = Database(settings.db_path)
    repo = Repository(db)
    index = EmbeddingIndex(repo, providers.embedder, settings.embed_batch_size) if providers.embedder else None
    search = SemanticSearch(repo, index, settings.min_score)
    tz = ZoneInfo(settings.timezone)
    scheduler = Scheduler(repo, search, settings.meeting_default_minutes,
                          now_fn=now_fn or (lambda: datetime.now(tz)))
    assistant = Assistant(repo, scheduler, providers.llm, settings.timezone, settings.history_messages,
                          settings.meeting_default_minutes)
    return AppContext(settings, db, repo, search, scheduler, assistant, providers)

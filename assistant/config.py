"""Настройки приложения: читаются из переменных окружения и файла .env.

PROVIDER=yandex (по умолчанию): YandexGPT, эмбеддинги и SpeechKit в Yandex Cloud (оплата в рублях).
PROVIDER=local: бесплатно и локально (Ollama + faster-whisper или браузер).
PROVIDER=openai: OpenAI API.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent


class ConfigError(RuntimeError):
    """Ошибка конфигурации, понятная пользователю (нет ключа, не установлен модуль и т.п.)."""


def _bool(value: str | None, default: bool) -> bool:
    if value is None or value.strip() == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "on", "да"}


def _path(value: str | None, default: Path) -> Path:
    if not value:
        return default
    p = Path(value)
    return p if p.is_absolute() else BASE_DIR / p


@dataclass
class Settings:
    provider: str = "yandex"                     # yandex | local | openai

    # Yandex Cloud (AI Studio + SpeechKit)
    yandex_api_key: str = ""                     # API-ключ сервисного аккаунта
    yandex_iam_token: str = ""                   # альтернатива ключу (IAM-токен живет до 12 часов)
    yandex_folder_id: str = ""
    yandex_llm_base_url: str = "https://llm.api.cloud.yandex.net/v1"
    yandex_gpt_model: str = "yandexgpt-5.1"      # YandexGPT Pro 5.1; дешевле - aliceai-llm-flash
    yandex_temperature: float = 0.1
    yandex_embed_doc_model: str = "text-search-doc/latest"
    yandex_embed_query_model: str = "text-search-query/latest"
    yandex_embed_rps: float = 8.0                # квота API - 10 запросов в секунду
    yandex_stt_url: str = "https://stt.api.cloud.yandex.net/speech/v1/stt:recognize"
    yandex_tts_url: str = "https://tts.api.cloud.yandex.net/speech/v1/tts:synthesize"
    yandex_tts_voice: str = "filipp"
    yandex_tts_speed: float = 1.0

    # Локальная LLM и эмбеддинги (Ollama, OpenAI-совместимый API)
    llm_base_url: str = "http://localhost:11434/v1"
    llm_model: str = "qwen2.5:7b"
    llm_api_key: str = "ollama"                  # Ollama ключ не проверяет
    llm_tools_mode: str = "auto"                 # auto | tools | json
    llm_temperature: float = 0.0
    embed_base_url: str = ""                     # пусто = как у LLM
    embed_model: str = "bge-m3"
    embed_batch_size: int = 32
    search_min_score: float | None = None        # порог сходства; по умолчанию зависит от модели

    # Речь
    stt_engine: str = "auto"                     # auto | yandex | local | openai | browser
    whisper_model: str = "small"
    whisper_device: str = "cpu"
    whisper_compute_type: str = "int8"
    stt_language: str = "ru"
    tts_engine: str = "auto"                     # auto | yandex | browser | openai

    # OpenAI (опционально)
    openai_api_key: str = ""
    openai_base_url: str | None = None
    openai_chat_model: str = "gpt-4o-mini"
    openai_stt_model: str = "whisper-1"
    openai_tts_model: str = "gpt-4o-mini-tts"
    openai_tts_voice: str = "alloy"
    openai_embed_model: str = "text-embedding-3-small"

    # Приложение
    timezone: str = "Europe/Moscow"
    db_path: Path = field(default_factory=lambda: BASE_DIR / "data" / "assistant.db")
    default_user: str = "Основной"
    import_csv_on_start: bool = True
    history_csv: Path = field(
        default_factory=lambda: BASE_DIR / "data" / "voice_assistant_events_1000_77e927b5d2.csv"
    )
    meeting_default_minutes: int = 60
    history_messages: int = 10
    host: str = "127.0.0.1"
    port: int = 8000

    # ---------------------------------------------------------------- производные
    @property
    def is_openai(self) -> bool:
        return self.provider == "openai"

    @property
    def is_yandex(self) -> bool:
        return self.provider == "yandex"

    @property
    def yandex_configured(self) -> bool:
        return bool((self.yandex_api_key or self.yandex_iam_token).strip() and self.yandex_folder_id.strip())

    @property
    def yandex_gpt_uri(self) -> str:
        m = self.yandex_gpt_model.strip()
        return m if m.startswith("gpt://") else f"gpt://{self.yandex_folder_id}/{m}"

    def yandex_emb_uri(self, model: str) -> str:
        return model if model.startswith("emb://") else f"emb://{self.yandex_folder_id}/{model}"

    @property
    def openai_configured(self) -> bool:
        return bool(self.openai_api_key.strip())

    @property
    def chat_model(self) -> str:
        if self.is_yandex:
            return self.yandex_gpt_model
        return self.openai_chat_model if self.is_openai else self.llm_model

    @property
    def active_embed_model(self) -> str:
        if self.is_yandex:
            return f"{self.yandex_embed_doc_model} / {self.yandex_embed_query_model}"
        return self.openai_embed_model if self.is_openai else self.embed_model

    @property
    def min_score(self) -> float:
        if self.search_min_score is not None:
            return self.search_min_score
        if self.is_yandex:
            return 0.35
        return 0.3 if self.is_openai else 0.45  # у bge-m3 сходство несвязанных фраз выше

    @property
    def resolved_stt(self) -> str:
        if self.stt_engine in ("yandex", "local", "openai", "browser"):
            return self.stt_engine
        return {"yandex": "yandex", "openai": "openai"}.get(self.provider, "local")

    @property
    def resolved_tts(self) -> str:
        if self.tts_engine in ("yandex", "browser", "openai"):
            return self.tts_engine
        return {"yandex": "yandex", "openai": "openai"}.get(self.provider, "browser")

    @classmethod
    def from_env(cls, env_file: Path | None = None) -> "Settings":
        try:
            from dotenv import load_dotenv

            load_dotenv(env_file or BASE_DIR / ".env", override=False)
        except ImportError:  # pragma: no cover
            pass
        e = os.environ.get
        d = cls()
        return cls(
            provider=(e("PROVIDER") or d.provider).strip().lower(),
            yandex_api_key=(e("YANDEX_API_KEY") or "").strip(),
            yandex_iam_token=(e("YANDEX_IAM_TOKEN") or "").strip(),
            yandex_folder_id=(e("YANDEX_FOLDER_ID") or "").strip(),
            yandex_llm_base_url=e("YANDEX_LLM_BASE_URL") or d.yandex_llm_base_url,
            yandex_gpt_model=e("YANDEX_GPT_MODEL") or d.yandex_gpt_model,
            yandex_temperature=float(e("YANDEX_TEMPERATURE") or d.yandex_temperature),
            yandex_embed_doc_model=e("YANDEX_EMBED_DOC_MODEL") or d.yandex_embed_doc_model,
            yandex_embed_query_model=e("YANDEX_EMBED_QUERY_MODEL") or d.yandex_embed_query_model,
            yandex_embed_rps=float(e("YANDEX_EMBED_RPS") or d.yandex_embed_rps),
            yandex_stt_url=e("YANDEX_STT_URL") or d.yandex_stt_url,
            yandex_tts_url=e("YANDEX_TTS_URL") or d.yandex_tts_url,
            yandex_tts_voice=e("YANDEX_TTS_VOICE") or d.yandex_tts_voice,
            yandex_tts_speed=float(e("YANDEX_TTS_SPEED") or d.yandex_tts_speed),
            llm_base_url=e("LLM_BASE_URL") or d.llm_base_url,
            llm_model=e("LLM_MODEL") or d.llm_model,
            llm_api_key=e("LLM_API_KEY") or d.llm_api_key,
            llm_tools_mode=(e("LLM_TOOLS_MODE") or d.llm_tools_mode).strip().lower(),
            llm_temperature=float(e("LLM_TEMPERATURE") or d.llm_temperature),
            embed_base_url=e("EMBED_BASE_URL") or d.embed_base_url,
            embed_model=e("EMBED_MODEL") or d.embed_model,
            embed_batch_size=int(e("EMBED_BATCH_SIZE") or d.embed_batch_size),
            search_min_score=float(e("SEARCH_MIN_SCORE")) if e("SEARCH_MIN_SCORE") else None,
            stt_engine=(e("STT_ENGINE") or d.stt_engine).strip().lower(),
            whisper_model=e("WHISPER_MODEL") or d.whisper_model,
            whisper_device=e("WHISPER_DEVICE") or d.whisper_device,
            whisper_compute_type=e("WHISPER_COMPUTE_TYPE") or d.whisper_compute_type,
            stt_language=e("STT_LANGUAGE") or d.stt_language,
            tts_engine=(e("TTS_ENGINE") or d.tts_engine).strip().lower(),
            openai_api_key=(e("OPENAI_API_KEY") or "").strip(),
            openai_base_url=(e("OPENAI_BASE_URL") or "").strip() or None,
            openai_chat_model=e("OPENAI_CHAT_MODEL") or d.openai_chat_model,
            openai_stt_model=e("OPENAI_STT_MODEL") or d.openai_stt_model,
            openai_tts_model=e("OPENAI_TTS_MODEL") or d.openai_tts_model,
            openai_tts_voice=e("OPENAI_TTS_VOICE") or d.openai_tts_voice,
            openai_embed_model=e("OPENAI_EMBED_MODEL") or d.openai_embed_model,
            timezone=e("APP_TIMEZONE") or d.timezone,
            db_path=_path(e("DB_PATH"), d.db_path),
            default_user=e("DEFAULT_USER") or d.default_user,
            import_csv_on_start=_bool(e("IMPORT_CSV_ON_START"), d.import_csv_on_start),
            history_csv=_path(e("HISTORY_CSV"), d.history_csv),
            meeting_default_minutes=int(e("MEETING_DEFAULT_MINUTES") or d.meeting_default_minutes),
            history_messages=int(e("HISTORY_MESSAGES") or d.history_messages),
            host=e("HOST") or d.host,
            port=int(e("PORT") or d.port),
        )

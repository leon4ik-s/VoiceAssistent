"""Голосовой слой: распознавание речи (STT) и озвучка ответов (TTS).

По умолчанию речь распознается локально через faster-whisper, а озвучка идет в браузере
(speechSynthesis), поэтому серверный TTS нужен только при TTS_ENGINE=openai.
Запасной вариант распознавания без установки - Web Speech API в браузере (Chrome).
"""
from __future__ import annotations

import io
import logging
import threading
from typing import Protocol

log = logging.getLogger(__name__)

MAX_TTS_CHARS = 4000
EXT_BY_MIME = {
    "audio/webm": "webm",
    "audio/ogg": "ogg",
    "audio/mp4": "mp4",
    "audio/x-m4a": "m4a",
    "audio/m4a": "m4a",
    "audio/mpeg": "mp3",
    "audio/wav": "wav",
    "audio/x-wav": "wav",
}


class SpeechToText(Protocol):
    def transcribe(self, audio: bytes, filename: str = "audio.webm", content_type: str | None = None) -> str: ...


class TextToSpeech(Protocol):
    def synthesize(self, text: str) -> bytes: ...


def guess_filename(filename: str | None, content_type: str | None) -> str:
    if filename and "." in filename:
        return filename
    base = (content_type or "").split(";")[0].strip().lower()
    return f"speech.{EXT_BY_MIME.get(base, 'webm')}"


class FasterWhisperSTT:
    """Локальное распознавание. Модель скачивается автоматически при первом вызове (small ~ 500 МБ)."""

    def __init__(self, model: str = "small", device: str = "cpu", compute_type: str = "int8", language: str = "ru"):
        self.model_name = model
        self.device = device
        self.compute_type = compute_type
        self.language = language
        self._model = None
        self._lock = threading.Lock()

    def _load(self):
        with self._lock:
            if self._model is None:
                from faster_whisper import WhisperModel

                log.info("Загрузка модели faster-whisper %s (%s, %s)...", self.model_name, self.device, self.compute_type)
                try:
                    self._model = WhisperModel(self.model_name, device=self.device, compute_type=self.compute_type)
                except Exception as exc:
                    from ..config import ConfigError

                    raise ConfigError(
                        f"Не удалось загрузить модель распознавания «{self.model_name}» ({type(exc).__name__}). "
                        "При первом запуске модель скачивается из интернета (small - около 500 МБ). "
                        "Пока можно распознавать речь в браузере (Chrome)."
                    ) from exc
        return self._model

    def transcribe(self, audio: bytes, filename: str = "audio.webm", content_type: str | None = None) -> str:
        model = self._load()
        segments, _info = model.transcribe(io.BytesIO(audio), language=self.language or None,
                                           beam_size=5, vad_filter=True)
        return " ".join(seg.text.strip() for seg in segments).strip()


class OpenAISpeechToText:
    def __init__(self, client, model: str = "whisper-1", language: str = "ru"):
        self.client = client
        self.model = model
        self.language = language

    def transcribe(self, audio: bytes, filename: str = "audio.webm", content_type: str | None = None) -> str:
        name = guess_filename(filename, content_type)
        result = self.client.audio.transcriptions.create(
            model=self.model,
            file=(name, audio, content_type or "application/octet-stream"),
            language=self.language,
            response_format="text",
        )
        text = result if isinstance(result, str) else getattr(result, "text", "")
        return (text or "").strip()


class OpenAITextToSpeech:
    def __init__(self, client, model: str = "gpt-4o-mini-tts", voice: str = "alloy"):
        self.client = client
        self.model = model
        self.voice = voice

    def synthesize(self, text: str) -> bytes:
        resp = self.client.audio.speech.create(
            model=self.model,
            voice=self.voice,
            input=text[:MAX_TTS_CHARS],
            response_format="mp3",
        )
        return resp.content

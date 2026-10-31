"""Yandex SpeechKit API v1: синхронное распознавание коротких фраз и синтез речи.

Документация:
- распознавание: https://yandex.cloud/ru/docs/speechkit/stt/api/request-api
  POST https://stt.api.cloud.yandex.net/speech/v1/stt:recognize?lang=ru-RU&format=lpcm&sampleRateHertz=16000
  тело - аудио (lpcm или oggopus), до 1 МБ и 30 секунд, один канал; ответ {"result": "..."}.
- синтез: https://yandex.cloud/ru/docs/speechkit/tts/request
  POST https://tts.api.cloud.yandex.net/speech/v1/tts:synthesize, form-urlencoded: text, lang, voice, speed, format.

Браузер присылает WAV (16 кГц, моно, PCM) - его заголовок снимается и данные уходят как lpcm,
поэтому ffmpeg не нужен. OGG/Opus передается как есть. Другие форматы (webm, mp4) конвертируются
через ffmpeg, если он установлен.
"""
from __future__ import annotations

import io
import json
import shutil
import subprocess
import urllib.parse
import wave

import numpy as np

from ..config import ConfigError
from ..yandex_cloud import YandexAPIError, auth_header, http_post, parse_json_error

STT_RATES = (8000, 16000, 48000)
MAX_STT_BYTES = 1024 * 1024
MAX_STT_SECONDS = 30
MAX_TTS_CHARS = 2000  # лимит API 5000 символов и 15 КБ тела запроса; кириллица в URL-кодировке весит больше


def wav_to_lpcm(data: bytes) -> tuple[bytes, int]:
    """WAV -> (сырые 16-битные моно-сэмплы, частота). Стерео сводится в моно, нестандартная частота -> 16 кГц."""
    with wave.open(io.BytesIO(data), "rb") as w:
        channels, width, rate = w.getnchannels(), w.getsampwidth(), w.getframerate()
        frames = w.readframes(w.getnframes())
    if width != 2:
        raise ConfigError("Поддерживается только 16-битный WAV.")
    samples = np.frombuffer(frames, dtype="<i2").astype(np.float32)
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)
    if rate not in STT_RATES:
        n = int(len(samples) * 16000 / rate)
        samples = np.interp(np.linspace(0, len(samples) - 1, n), np.arange(len(samples)), samples)
        rate = 16000
    return np.clip(samples, -32768, 32767).astype("<i2").tobytes(), rate


def ffmpeg_to_lpcm(data: bytes) -> tuple[bytes, int]:
    exe = shutil.which("ffmpeg")
    if not exe:
        raise ConfigError("Этот формат записи SpeechKit не принимает, а ffmpeg не установлен. "
                          "Обновите страницу (она пишет звук в WAV) или установите ffmpeg.")
    proc = subprocess.run([exe, "-loglevel", "error", "-i", "pipe:0", "-ac", "1", "-ar", "16000", "-f", "s16le", "pipe:1"],
                          input=data, capture_output=True, timeout=60)
    if proc.returncode != 0:
        raise ConfigError("ffmpeg не смог прочитать запись: " + proc.stderr.decode("utf-8", "replace")[:200])
    return proc.stdout, 16000


def prepare_audio(data: bytes) -> tuple[bytes, dict]:
    """Привести запись к формату SpeechKit. Возвращает (тело запроса, параметры формата)."""
    if data[:4] == b"RIFF" and data[8:12] == b"WAVE":
        pcm, rate = wav_to_lpcm(data)
    elif data[:4] == b"OggS":
        return data, {"format": "oggopus"}
    else:
        pcm, rate = ffmpeg_to_lpcm(data)
    seconds = len(pcm) / (2 * rate)
    if seconds > MAX_STT_SECONDS:
        raise ConfigError(f"Фраза слишком длинная ({seconds:.0f} с): SpeechKit распознает до 30 секунд за раз.")
    return pcm, {"format": "lpcm", "sampleRateHertz": str(rate)}


class SpeechKitSTT:
    def __init__(self, api_key: str, folder_id: str = "", iam_token: str = "", language: str = "ru-RU",
                 url: str = "https://stt.api.cloud.yandex.net/speech/v1/stt:recognize"):
        self.api_key, self.iam_token, self.folder_id = api_key, iam_token, folder_id
        self.language = language if "-" in language else {"ru": "ru-RU", "en": "en-US"}.get(language, language)
        self.url = url

    def transcribe(self, audio: bytes, filename: str = "", content_type: str | None = None) -> str:
        body, fmt = prepare_audio(audio)
        if len(body) > MAX_STT_BYTES:
            raise ConfigError("Запись больше 1 МБ: SpeechKit принимает короткие фразы до 30 секунд.")
        params = {"lang": self.language, "topic": "general", **fmt}
        if not self.api_key and self.folder_id:  # folderId нужен только для IAM-токена пользователя
            params["folderId"] = self.folder_id
        headers = {**auth_header(self.api_key, self.iam_token), "Content-Type": "application/octet-stream"}
        try:
            raw = http_post(f"{self.url}?{urllib.parse.urlencode(params)}", body, headers, "SpeechKit (распознавание)")
        except YandexAPIError as exc:
            exc.text = parse_json_error(exc.text)
            raise
        return (json.loads(raw.decode("utf-8")).get("result") or "").strip()


class SpeechKitTTS:
    def __init__(self, api_key: str, folder_id: str = "", iam_token: str = "", voice: str = "filipp",
                 speed: float = 1.0, url: str = "https://tts.api.cloud.yandex.net/speech/v1/tts:synthesize"):
        self.api_key, self.iam_token, self.folder_id = api_key, iam_token, folder_id
        self.voice, self.speed, self.url = voice, speed, url

    def synthesize(self, text: str) -> bytes:
        form = {"text": text[:MAX_TTS_CHARS], "lang": "ru-RU", "voice": self.voice,
                "speed": str(self.speed), "format": "mp3"}
        if not self.api_key and self.folder_id:
            form["folderId"] = self.folder_id
        headers = {**auth_header(self.api_key, self.iam_token),
                   "Content-Type": "application/x-www-form-urlencoded"}
        try:
            return http_post(self.url, urllib.parse.urlencode(form).encode("ascii"), headers, "SpeechKit (синтез)")
        except YandexAPIError as exc:
            exc.text = parse_json_error(exc.text)
            raise

"""Общее для Yandex Cloud: авторизация, HTTP-запросы, понятные ошибки, эмбеддинги.

Документация:
- OpenAI-совместимый API AI Studio: https://yandex.cloud/ru/docs/ai-studio/concepts/openai-compatibility
- Аутентификация: заголовок «Authorization: Api-Key <ключ>» или «Authorization: Bearer <IAM-токен>».
"""
from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.request
from typing import Callable

log = logging.getLogger(__name__)


class YandexAPIError(RuntimeError):
    def __init__(self, status: int | None, text: str, service: str = "Yandex Cloud"):
        super().__init__(f"{service}: HTTP {status}: {text}")
        self.status_code = status
        self.text = text
        self.service = service


def auth_header(api_key: str, iam_token: str = "") -> dict:
    if api_key:
        return {"Authorization": f"Api-Key {api_key}"}
    return {"Authorization": f"Bearer {iam_token}"}


def http_post(url: str, data: bytes, headers: dict, service: str, timeout: float = 60) -> bytes:
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read()
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace")
        raise YandexAPIError(exc.code, body[:500], service) from None
    except urllib.error.URLError as exc:
        raise YandexAPIError(None, str(exc.reason), service) from None


BILLING_WORDS = ("billing", "payment", "balance", "suspended", "баланс", "платеж", "оплат")


def yandex_error_message(status: int | None, text: str = "", service: str = "Yandex Cloud") -> tuple[int, str]:
    """HTTP-статус и текст ошибки Yandex Cloud -> (код для интерфейса, понятное сообщение)."""
    low = (text or "").lower()
    if status is None:
        return 502, f"Нет связи с {service}. Проверьте интернет и повторите."
    if status == 402 or any(w in low for w in BILLING_WORDS):
        return 402, ("Yandex Cloud отклонил запрос из-за оплаты: платежный аккаунт не активен или на балансе нет средств. "
                     "Пополните баланс в консоли Yandex Cloud (раздел «Биллинг»).")
    if status == 401:
        return 401, "Yandex Cloud не принял ключ. Проверьте YANDEX_API_KEY в .env (ключ сервисного аккаунта, без пробелов)."
    if status == 403:
        return 403, (f"Нет доступа к {service}. Проверьте, что YANDEX_FOLDER_ID - это каталог сервисного аккаунта, "
                     "и что у аккаунта есть роль ai.editor или роли ai.languageModels.user, "
                     "ai.speechkit-stt.user, ai.speechkit-tts.user.")
    if status == 429:
        return 429, "Превышен лимит запросов Yandex Cloud. Подождите несколько секунд и повторите."
    if status == 404 or (status == 400 and ("model" in low or "uri" in low)):
        return 400, ("Модель недоступна: проверьте YANDEX_GPT_MODEL и YANDEX_FOLDER_ID в .env "
                     "(устаревшие модели отключаются, актуальный список - в документации AI Studio).")
    if status == 400:
        return 400, f"{service} отклонил запрос: {text[:300]}"
    return 502, f"Ошибка {service} (HTTP {status}): {text[:300]}"


class YandexEmbedder:
    """Эмбеддинги AI Studio через OpenAI-совместимый /v1/embeddings.

    Записи векторизуются моделью text-search-doc, поисковые запросы - text-search-query.
    API принимает одну строку за запрос (квота - 10 запросов в секунду), поэтому здесь
    ограничение частоты и повтор с паузой при 429.
    """

    def __init__(self, client, folder_id: str, doc_model: str, query_model: str, rps: float = 8.0,
                 max_retries: int = 4, sleep: Callable[[float], None] = time.sleep):
        self.client = client
        self.folder_id = folder_id
        self.doc_model = doc_model
        self.query_model_name = query_model
        # имена для кэша: без каталога, чтобы смена каталога не сбрасывала кэш
        self.model = f"yandex:{doc_model}"
        self.query_model = f"yandex:{query_model}"
        self.min_interval = 1.0 / max(rps, 0.1)
        self.max_retries = max_retries
        self.sleep = sleep
        self._last = 0.0

    def _uri(self, model: str) -> str:
        return model if model.startswith("emb://") else f"emb://{self.folder_id}/{model}"

    def _one(self, model: str, text: str) -> list[float]:
        delay = 1.0
        for attempt in range(self.max_retries + 1):
            wait = self.min_interval - (time.monotonic() - self._last)
            if wait > 0:
                self.sleep(wait)
            self._last = time.monotonic()
            try:
                resp = self.client.embeddings.create(model=self._uri(model), input=text, encoding_format="float")
                return resp.data[0].embedding
            except Exception as exc:
                if getattr(exc, "status_code", None) == 429 and attempt < self.max_retries:
                    log.info("Yandex embeddings: 429, пауза %.1f c", delay)
                    self.sleep(delay)
                    delay *= 2
                    continue
                raise
        raise RuntimeError("unreachable")  # pragma: no cover

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._one(self.doc_model, t) for t in texts]

    def embed_query(self, texts: list[str]) -> list[list[float]]:
        return [self._one(self.query_model_name, t) for t in texts]


def parse_json_error(raw: str) -> str:
    try:
        obj = json.loads(raw)
        return obj.get("error_message") or obj.get("message") or (obj.get("error") or {}).get("message") or raw
    except (ValueError, AttributeError):
        return raw

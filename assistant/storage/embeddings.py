"""Векторный индекс: эмбеддинги текстов с кэшем в SQLite.

Один и тот же текст для одной модели считается только один раз, поэтому
повторный импорт и повторный поиск не тратят запросы к API.
"""
from __future__ import annotations

import hashlib
import logging
import re
import threading
from typing import Protocol

import numpy as np

from .repository import Repository

log = logging.getLogger(__name__)


class Embedder(Protocol):
    model: str

    def embed(self, texts: list[str]) -> list[list[float]]: ...


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def cache_key(model: str, text: str) -> str:
    return hashlib.sha256(f"{model}\n{normalize_text(text)}".encode("utf-8")).hexdigest()


class EmbeddingIndex:
    def __init__(self, repo: Repository, embedder: Embedder, batch_size: int = 100):
        self.repo = repo
        self.embedder = embedder
        self.batch_size = max(1, batch_size)
        self.api_calls = 0  # для тестов и логов
        self.progress = {"running": False, "done": 0, "total": 0}
        self.bulk_lock = threading.Lock()  # фоновые пересчеты идут по одному

    def vectors_for(self, texts: list[str], max_missing: int | None = None,
                    query: bool = False) -> dict[str, np.ndarray]:
        """Вернуть {текст: нормированный вектор}; недостающие посчитать батчами и закэшировать.

        Если недостающих больше max_missing, они не считаются (вернутся только готовые векторы):
        так поиск не ждет, пока в фоне досчитывается большой импорт.
        """
        # Ключ кэша включает имя модели: смена модели = пересчет. У некоторых провайдеров
        # (Yandex) для поисковых запросов отдельная модель - embed_query/query_model.
        embed_fn = self.embedder.embed
        model = self.embedder.model
        if query and hasattr(self.embedder, "embed_query"):
            embed_fn, model = self.embedder.embed_query, getattr(self.embedder, "query_model", model)
        uniq = list(dict.fromkeys(normalize_text(t) for t in texts if t and t.strip()))
        keys = {t: cache_key(model, t) for t in uniq}
        cached = self.repo.get_vectors(list(keys.values()))
        missing = [t for t in uniq if keys[t] not in cached]
        if max_missing is not None and len(missing) > max_missing:
            missing = []
        bulk = len(missing) > self.batch_size
        if bulk:
            self.progress = {"running": True, "done": 0, "total": len(missing)}
            log.info("Считаю эмбеддинги (%s): %s текстов", model, len(missing))
        try:
            for i in range(0, len(missing), self.batch_size):
                batch = missing[i : i + self.batch_size]
                vecs = embed_fn(batch)
                self.api_calls += 1
                fresh = {keys[t]: np.asarray(v, dtype=np.float32) for t, v in zip(batch, vecs)}
                self.repo.put_vectors(model, fresh)
                cached.update(fresh)
                if bulk:
                    self.progress["done"] = min(len(missing), i + len(batch))
                    log.info("Эмбеддинги: %s из %s", self.progress["done"], len(missing))
        finally:
            if bulk:
                self.progress["running"] = False
        out = {}
        for t in uniq:
            if keys[t] not in cached:
                continue
            v = cached[keys[t]]
            n = float(np.linalg.norm(v))
            out[t] = v / n if n else v
        return out

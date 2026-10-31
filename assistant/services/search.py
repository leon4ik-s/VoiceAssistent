"""Поиск записей по смыслу: эмбеддинги + простое совпадение основ слов.

Если ключ OpenAI не задан, работает только словесная часть поиска.
"""
from __future__ import annotations

import logging
import re
from datetime import date

import numpy as np

from ..storage.embeddings import EmbeddingIndex, normalize_text
from ..storage.models import Item
from ..storage.repository import Repository
from . import recurrence

log = logging.getLogger(__name__)

STOP = {
    "в", "на", "с", "со", "по", "у", "что", "когда", "было", "была", "был", "были", "меня", "мне",
    "для", "и", "о", "об", "про", "к", "ко", "найди", "найти", "это", "все", "всё", "мой", "моя",
    "мои", "какие", "какая", "какой", "где", "а", "ли", "до", "от", "за", "из", "же", "там",
}
SEMANTIC_WEIGHT = 0.8
MIN_SEMANTIC_SCORE = 0.25
MAX_SYNC_EMBED = 50  # больше - значит идет большой импорт, его досчитывает фоновый поток


def tokens(text: str) -> set[str]:
    words = re.findall(r"[a-zа-я0-9]+", text.lower().replace("ё", "е"))
    return {w for w in words if w not in STOP and len(w) > 1}


def stems(text: str) -> set[str]:
    """Грубые основы слов (первые 5 букв) - для словесной части поиска без морфологии."""
    return {w[:5] for w in tokens(text)}


def _same_root(a: str, b: str) -> bool:
    n = min(5, len(a), len(b))
    return n >= 3 and a[:n] == b[:n]


def lexical_score(query: str | set[str], text: str) -> float:
    """Доля слов запроса, которые встречаются в тексте (с учетом окончаний: Иван/Ивану/Иваном)."""
    q = tokens(query) if isinstance(query, str) else query
    if not q:
        return 0.0
    d = tokens(text)
    return sum(1 for w in q if any(_same_root(w, x) for x in d)) / len(q)


def in_range(item: Item, date_from: date | None, date_to: date | None) -> bool:
    if not date_from and not date_to:
        return True
    lo, hi = date_from or date.min, date_to or date.max
    if not item.recurrence:
        return lo <= item.start_date <= hi
    end = recurrence.last_date(item)
    return item.start_date <= hi and (end is None or end >= lo)


class SemanticSearch:
    def __init__(self, repo: Repository, index: EmbeddingIndex | None = None,
                 min_score: float = MIN_SEMANTIC_SCORE):
        self.repo = repo
        self.index = index
        self.min_score = min_score

    def ensure_embeddings(self, user_id: int) -> int:
        """Досчитать эмбеддинги для всех записей пользователя. Возвращает число записей."""
        if not self.index:
            return 0
        with self.index.bulk_lock:
            items = self.repo.list_items(user_id)
            self.index.vectors_for([i.title for i in items])
        return len(items)

    def search(
        self,
        user_id: int,
        query: str,
        limit: int = 10,
        date_from: date | None = None,
        date_to: date | None = None,
        kinds: list[str] | None = None,
        items: list[Item] | None = None,
    ) -> list[tuple[Item, float]]:
        if items is None:
            items = self.repo.list_items(user_id, kinds=kinds)
        items = [i for i in items if in_range(i, date_from, date_to) and (not kinds or i.kind in kinds)]
        if not items or not query.strip():
            return []
        q_tokens = tokens(query)
        lex = np.array([lexical_score(q_tokens, i.title) for i in items], dtype=np.float32)

        sem = None
        if self.index is not None:
            try:
                qv = self.index.vectors_for([query], query=True)[normalize_text(query)]
                vecs = self.index.vectors_for([i.title for i in items], max_missing=MAX_SYNC_EMBED)
                zero = np.zeros_like(qv)
                mat = np.stack([vecs.get(normalize_text(i.title), zero) for i in items])
                sem = mat @ qv
            except Exception as exc:  # сеть, лимиты и т.п. - деградируем до словесного поиска
                log.warning("Семантический поиск недоступен: %s", exc)
                sem = None

        if sem is not None:
            scores = SEMANTIC_WEIGHT * sem + (1 - SEMANTIC_WEIGHT) * lex
            keep = (sem >= self.min_score) | (lex > 0)
        else:
            scores = lex
            keep = lex > 0
        order = np.argsort(-scores, kind="stable")
        out = [(items[k], float(scores[k])) for k in order if keep[k]]
        return out[:limit]

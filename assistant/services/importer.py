"""Импорт истории прошлых запросов из CSV (колонки id,date,time,text).

Импорт идемпотентный: каждая строка получает external_id="csv:<id>",
повторный запуск не создает дублей.
"""
from __future__ import annotations

import csv
import logging
from pathlib import Path

from ..storage.models import Item
from ..storage.repository import Repository
from .scheduler import parse_date, parse_time

log = logging.getLogger(__name__)

MEETING_PREFIXES = ("встреча", "созвон", "демо", "провести планёрку", "провести планерку", "планёрка",
                    "планерка", "совещание", "звонок с")
REMINDER_PREFIXES = ("напомнить", "напоминание")
SOURCE = "import"


def classify(text: str) -> str:
    t = text.strip().lower()
    if t.startswith(MEETING_PREFIXES):
        return "meeting"
    if t.startswith(REMINDER_PREFIXES):
        return "reminder"
    return "task"


def read_csv(path: Path) -> list[dict]:
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def import_csv(repo: Repository, user_id: int, path: str | Path, meeting_minutes: int = 60) -> dict:
    path = Path(path)
    rows = read_csv(path)
    items, bad = [], 0
    for n, row in enumerate(rows, start=1):
        try:
            text = (row.get("text") or "").strip()
            if not text:
                raise ValueError("пустой текст")
            kind = classify(text)
            items.append(Item(
                user_id=user_id,
                kind=kind,
                title=text,
                date=parse_date(row["date"]),
                time=parse_time(row["time"]) if (row.get("time") or "").strip() else None,
                duration_min=meeting_minutes if kind == "meeting" else None,
                source=SOURCE,
                external_id=f"csv:{(row.get('id') or n)}".strip(),
            ))
        except (KeyError, ValueError) as exc:
            bad += 1
            log.warning("Строка %s пропущена: %s", n, exc)
    inserted = repo.add_items_ignore_existing(items)
    return {"rows": len(rows), "inserted": inserted, "skipped_existing": len(items) - inserted, "invalid": bad}

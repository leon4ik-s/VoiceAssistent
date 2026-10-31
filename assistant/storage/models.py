"""Модели данных."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date, datetime, time
from typing import Any

KINDS = ("task", "reminder", "meeting")
KIND_RU = {"task": "задача", "reminder": "напоминание", "meeting": "встреча"}


@dataclass
class Item:
    user_id: int
    kind: str
    title: str
    date: str
    time: str | None = None
    duration_min: int | None = None
    status: str = "open"
    recurrence: dict | None = None
    done_dates: list[str] = field(default_factory=list)
    source: str = "user"
    external_id: str | None = None
    id: int | None = None
    created_at: str | None = None
    updated_at: str | None = None

    @classmethod
    def from_row(cls, row: Any) -> "Item":
        return cls(
            id=row["id"],
            user_id=row["user_id"],
            kind=row["kind"],
            title=row["title"],
            date=row["date"],
            time=row["time"],
            duration_min=row["duration_min"],
            status=row["status"],
            recurrence=json.loads(row["recurrence"]) if row["recurrence"] else None,
            done_dates=json.loads(row["done_dates"] or "[]"),
            source=row["source"],
            external_id=row["external_id"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    @property
    def start_date(self) -> date:
        return date.fromisoformat(self.date)

    @property
    def start_time(self) -> time | None:
        return time.fromisoformat(self.time) if self.time else None

    def start_dt(self) -> datetime:
        return datetime.combine(self.start_date, self.start_time or time(0, 0))

    def to_dict(self) -> dict:
        d = {
            "id": self.id,
            "kind": self.kind,
            "kind_ru": KIND_RU.get(self.kind, self.kind),
            "title": self.title,
            "date": self.date,
            "time": self.time,
            "status": self.status,
        }
        if self.kind == "meeting":
            d["duration_min"] = self.duration_min
        if self.recurrence:
            d["recurrence"] = self.recurrence
        if self.source != "user":
            d["source"] = self.source
        return d

    def short(self) -> str:
        when = self.date + (f" {self.time}" if self.time else "")
        rep = " (повтор)" if self.recurrence else ""
        done = " [выполнено]" if self.status == "done" else ""
        return f"#{self.id} {KIND_RU.get(self.kind, self.kind)} «{self.title}» {when}{rep}{done}"

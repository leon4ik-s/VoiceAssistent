"""Репозиторий: все SQL-запросы приложения собраны здесь."""
from __future__ import annotations

import json
from datetime import datetime
from typing import Iterable

import numpy as np

from .db import Database
from .models import Item

ITEM_FIELDS = ("kind", "title", "date", "time", "duration_min", "status", "recurrence", "done_dates")


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


class Repository:
    def __init__(self, db: Database):
        self.db = db

    # ---------- пользователи ----------
    def list_users(self) -> list[dict]:
        return [dict(r) for r in self.db.query("SELECT id, name FROM users ORDER BY id")]

    def get_user(self, user_id: int) -> dict | None:
        rows = self.db.query("SELECT id, name FROM users WHERE id = ?", (user_id,))
        return dict(rows[0]) if rows else None

    def get_or_create_user(self, name: str) -> dict:
        name = name.strip()
        rows = self.db.query("SELECT id, name FROM users WHERE name = ?", (name,))
        if rows:
            return dict(rows[0])
        cur = self.db.execute("INSERT INTO users (name, created_at) VALUES (?, ?)", (name, _now()))
        return {"id": cur.lastrowid, "name": name}

    # ---------- записи ----------
    def add_item(self, item: Item) -> Item:
        now = _now()
        cur = self.db.execute(
            """INSERT INTO items (user_id, kind, title, date, time, duration_min, status,
                                  recurrence, done_dates, source, external_id, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                item.user_id, item.kind, item.title, item.date, item.time, item.duration_min,
                item.status, json.dumps(item.recurrence, ensure_ascii=False) if item.recurrence else None,
                json.dumps(item.done_dates), item.source, item.external_id, now, now,
            ),
        )
        item.id, item.created_at, item.updated_at = cur.lastrowid, now, now
        return item

    def add_items_ignore_existing(self, items: Iterable[Item]) -> int:
        """Массовая вставка; записи с тем же (user_id, source, external_id) пропускаются."""
        now = _now()
        inserted = 0
        with self.db.transaction() as conn:
            for it in items:
                cur = conn.execute(
                    """INSERT OR IGNORE INTO items (user_id, kind, title, date, time, duration_min, status,
                           recurrence, done_dates, source, external_id, created_at, updated_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        it.user_id, it.kind, it.title, it.date, it.time, it.duration_min, it.status,
                        json.dumps(it.recurrence, ensure_ascii=False) if it.recurrence else None,
                        json.dumps(it.done_dates), it.source, it.external_id, now, now,
                    ),
                )
                inserted += cur.rowcount
        return inserted

    def find_by_external(self, user_id: int, source: str, external_id: str) -> Item | None:
        rows = self.db.query(
            "SELECT * FROM items WHERE user_id = ? AND source = ? AND external_id = ?",
            (user_id, source, external_id),
        )
        return Item.from_row(rows[0]) if rows else None

    def get_item(self, user_id: int, item_id: int) -> Item | None:
        rows = self.db.query("SELECT * FROM items WHERE id = ? AND user_id = ?", (item_id, user_id))
        return Item.from_row(rows[0]) if rows else None

    def get_items(self, user_id: int, ids: list[int]) -> list[Item]:
        out = []
        for i in ids:
            it = self.get_item(user_id, i)
            if it:
                out.append(it)
        return out

    def update_item(self, user_id: int, item_id: int, **fields) -> Item | None:
        sets, params = [], []
        for k, v in fields.items():
            if k not in ITEM_FIELDS:
                raise ValueError(f"Нельзя менять поле {k}")
            if k == "recurrence":
                v = json.dumps(v, ensure_ascii=False) if v else None
            if k == "done_dates":
                v = json.dumps(v)
            sets.append(f"{k} = ?")
            params.append(v)
        if sets:
            sets.append("updated_at = ?")
            params += [_now(), item_id, user_id]
            self.db.execute(f"UPDATE items SET {', '.join(sets)} WHERE id = ? AND user_id = ?", params)
        return self.get_item(user_id, item_id)

    def delete_item(self, user_id: int, item_id: int) -> bool:
        cur = self.db.execute("DELETE FROM items WHERE id = ? AND user_id = ?", (item_id, user_id))
        return cur.rowcount > 0

    def list_items(
        self,
        user_id: int,
        date_to: str | None = None,
        kinds: Iterable[str] | None = None,
        source: str | None = None,
    ) -> list[Item]:
        sql, params = "SELECT * FROM items WHERE user_id = ?", [user_id]
        if date_to:
            sql += " AND date <= ?"
            params.append(date_to)
        if kinds:
            kinds = list(kinds)
            sql += f" AND kind IN ({','.join('?' * len(kinds))})"
            params += kinds
        if source:
            sql += " AND source = ?"
            params.append(source)
        sql += " ORDER BY date, COALESCE(time, '00:00'), id"
        return [Item.from_row(r) for r in self.db.query(sql, params)]

    def count_items(self, user_id: int, source: str | None = None) -> int:
        sql, params = "SELECT COUNT(*) FROM items WHERE user_id = ?", [user_id]
        if source:
            sql += " AND source = ?"
            params.append(source)
        return self.db.query(sql, params)[0][0]

    # ---------- история диалога ----------
    def add_message(self, user_id: int, role: str, content: str) -> None:
        self.db.execute(
            "INSERT INTO messages (user_id, role, content, created_at) VALUES (?, ?, ?, ?)",
            (user_id, role, content, _now()),
        )

    def recent_messages(self, user_id: int, limit: int) -> list[dict]:
        rows = self.db.query(
            "SELECT role, content, created_at FROM messages WHERE user_id = ? ORDER BY id DESC LIMIT ?",
            (user_id, limit),
        )
        return [dict(r) for r in reversed(rows)]

    # ---------- контекст диалога ----------
    def get_recent_ids(self, user_id: int) -> list[int]:
        rows = self.db.query("SELECT recent_item_ids FROM dialog_state WHERE user_id = ?", (user_id,))
        return json.loads(rows[0][0]) if rows else []

    def set_recent_ids(self, user_id: int, ids: list[int]) -> None:
        self.db.execute(
            """INSERT INTO dialog_state (user_id, recent_item_ids, updated_at) VALUES (?, ?, ?)
               ON CONFLICT(user_id) DO UPDATE SET recent_item_ids = excluded.recent_item_ids,
                                                  updated_at = excluded.updated_at""",
            (user_id, json.dumps(ids), _now()),
        )

    # ---------- кэш эмбеддингов ----------
    def get_vectors(self, keys: list[str]) -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {}
        for i in range(0, len(keys), 500):
            chunk = keys[i : i + 500]
            rows = self.db.query(
                f"SELECT key, vector FROM embedding_cache WHERE key IN ({','.join('?' * len(chunk))})", chunk
            )
            for r in rows:
                out[r["key"]] = np.frombuffer(r["vector"], dtype=np.float32)
        return out

    def put_vectors(self, model: str, vectors: dict[str, np.ndarray]) -> None:
        with self.db.transaction() as conn:
            for key, vec in vectors.items():
                v = np.asarray(vec, dtype=np.float32)
                conn.execute(
                    "INSERT OR REPLACE INTO embedding_cache (key, model, dim, vector) VALUES (?, ?, ?, ?)",
                    (key, model, int(v.shape[0]), v.tobytes()),
                )

    def count_vectors(self) -> int:
        return self.db.query("SELECT COUNT(*) FROM embedding_cache")[0][0]

"""Подключение к SQLite и схема БД."""
from __future__ import annotations

import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT NOT NULL UNIQUE,
    created_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS items (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id       INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    kind          TEXT NOT NULL CHECK (kind IN ('task', 'reminder', 'meeting')),
    title         TEXT NOT NULL,
    date          TEXT NOT NULL,              -- YYYY-MM-DD, местное время пользователя
    time          TEXT,                       -- HH:MM или NULL (задача на весь день)
    duration_min  INTEGER,                    -- длительность встречи
    status        TEXT NOT NULL DEFAULT 'open' CHECK (status IN ('open', 'done')),
    recurrence    TEXT,                       -- JSON-правило повтора или NULL
    done_dates    TEXT NOT NULL DEFAULT '[]', -- выполненные экземпляры повторяющейся задачи
    source        TEXT NOT NULL DEFAULT 'user', -- user | import | ics
    external_id   TEXT,                       -- id строки CSV или UID из .ics
    created_at    TEXT NOT NULL,
    updated_at    TEXT NOT NULL,
    UNIQUE (user_id, source, external_id)
);
CREATE INDEX IF NOT EXISTS ix_items_user_date ON items (user_id, date);

CREATE TABLE IF NOT EXISTS messages (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id     INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    role        TEXT NOT NULL,
    content     TEXT NOT NULL,
    created_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_messages_user ON messages (user_id, id);

CREATE TABLE IF NOT EXISTS dialog_state (
    user_id          INTEGER PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
    recent_item_ids  TEXT NOT NULL DEFAULT '[]',
    updated_at       TEXT
);

CREATE TABLE IF NOT EXISTS embedding_cache (
    key     TEXT PRIMARY KEY,   -- sha256(модель + текст)
    model   TEXT NOT NULL,
    dim     INTEGER NOT NULL,
    vector  BLOB NOT NULL       -- float32
);
"""


class Database:
    """Тонкая обертка над sqlite3 с блокировкой для работы из потоков FastAPI."""

    def __init__(self, path: str | Path):
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        if self.path != ":memory:":
            self._conn.execute("PRAGMA journal_mode = WAL")
        self._lock = threading.RLock()
        with self._lock:
            self._conn.executescript(SCHEMA)
            self._conn.commit()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            try:
                yield self._conn
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise

    def execute(self, sql: str, params: Any = ()) -> sqlite3.Cursor:
        with self.transaction() as conn:
            return conn.execute(sql, params)

    def query(self, sql: str, params: Any = ()) -> list[sqlite3.Row]:
        with self._lock:
            return self._conn.execute(sql, params).fetchall()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

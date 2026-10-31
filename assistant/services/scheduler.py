"""Бизнес-логика: создание, правка, удаление, выполнение, расписание, конфликты.

Методы возвращают словари со статусом:
  ok                  - действие выполнено;
  need_clarification  - не хватает данных (поле missing), нужно переспросить;
  ambiguous           - под описание подходит несколько записей (поле candidates);
  conflict            - встреча пересекается с другими (поле conflicts), нужен force=true;
  not_found / error   - запись не найдена или данные некорректны.
"""
from __future__ import annotations

import re
from dataclasses import replace
from datetime import date, datetime, time, timedelta
from typing import Callable

from ..storage.models import KINDS, KIND_RU, Item
from ..storage.repository import Repository
from . import recurrence
from .search import SemanticSearch, lexical_score

FIELD_RU = {"title": "название", "date": "дата", "time": "время", "kind": "тип записи"}
MAX_PERIOD_DAYS = 366
CONFLICT_HORIZON_DAYS = 90


class ValidationError(ValueError):
    pass


def parse_date(value) -> str:
    try:
        return date.fromisoformat(str(value).strip()[:10]).isoformat()
    except ValueError as exc:
        raise ValidationError(f"Некорректная дата: {value!r}, нужен формат YYYY-MM-DD") from exc


def parse_time(value) -> str:
    m = re.fullmatch(r"(\d{1,2})[:.](\d{2})(?::\d{2})?", str(value).strip())
    if not m or int(m.group(1)) > 23 or int(m.group(2)) > 59:
        raise ValidationError(f"Некорректное время: {value!r}, нужен формат HH:MM")
    return f"{int(m.group(1)):02d}:{m.group(2)}"


def _clarify(missing: list[str], hint: str = "") -> dict:
    names = ", ".join(FIELD_RU.get(m, m) for m in missing)
    return {
        "status": "need_clarification",
        "missing": missing,
        "message": f"Не хватает данных: {names}. Уточните у пользователя. {hint}".strip(),
    }


class Scheduler:
    def __init__(
        self,
        repo: Repository,
        search: SemanticSearch,
        meeting_default_minutes: int = 60,
        now_fn: Callable[[], datetime] | None = None,
    ):
        self.repo = repo
        self.search_service = search
        self.meeting_default_minutes = meeting_default_minutes
        self._now_fn = now_fn or datetime.now

    def now(self) -> datetime:
        return self._now_fn().replace(tzinfo=None, second=0, microsecond=0)

    # ------------------------------------------------------------------ helpers
    def view(self, item: Item, on_date: date | None = None) -> dict:
        d = item.to_dict()
        if item.recurrence:
            d["recurrence_ru"] = recurrence.describe(item.recurrence)
        if on_date is not None:
            d["date"] = on_date.isoformat()
            if item.recurrence and on_date.isoformat() in item.done_dates:
                d["status"] = "done"
        if item.kind == "meeting" and item.time:
            start = datetime.combine(on_date or item.start_date, item.start_time)
            d["end_time"] = (start + timedelta(minutes=item.duration_min or self.meeting_default_minutes)).strftime("%H:%M")
        return d

    def _interval(self, item: Item, on: date) -> tuple[datetime, datetime]:
        start = datetime.combine(on, item.start_time or time(0, 0))
        return start, start + timedelta(minutes=item.duration_min or self.meeting_default_minutes)

    def find_conflicts(self, user_id: int, item: Item, exclude_id: int | None = None) -> list[dict]:
        """Встречи пользователя, пересекающиеся по времени с item (для повторов - на 90 дней вперед)."""
        if item.kind != "meeting" or not item.time:
            return []
        start = item.start_date
        end = start
        if item.recurrence:
            last = recurrence.last_date(item)
            end = min(last or start + timedelta(days=CONFLICT_HORIZON_DAYS), start + timedelta(days=CONFLICT_HORIZON_DAYS))
        my_dates = recurrence.occurrence_dates(item, start, end)
        found = []
        for other in self.repo.list_items(user_id, date_to=end.isoformat(), kinds=["meeting"]):
            if other.id == exclude_id or not other.time or other.status == "done":
                continue
            other_dates = set(recurrence.occurrence_dates(other, start, end))
            for d in my_dates:
                if d not in other_dates:
                    continue
                a1, a2 = self._interval(item, d)
                b1, b2 = self._interval(other, d)
                if a1 < b2 and b1 < a2:
                    found.append(self.view(other, d))
                    break
        return found

    def _items_at(self, user_id: int, on_date: str | None, at_time: str | None,
                  kinds: list[str] | None) -> list[Item]:
        """Записи на указанные дату и время, с учетом повторов."""
        day = date.fromisoformat(parse_date(on_date)) if on_date else None
        clock = parse_time(at_time) if at_time else None
        return [i for i in self.repo.list_items(user_id, kinds=kinds)
                if (not clock or i.time == clock) and (not day or recurrence.occurrence_dates(i, day, day))]

    def _ambiguous(self, items: list[Item]) -> dict:
        return {
            "status": "ambiguous",
            "candidates": [self.view(i) for i in items[:5]],
            "message": "Под описание подходит несколько записей. Спросите пользователя, какую именно, "
            "и повторите вызов с item_id.",
        }

    def resolve(
        self,
        user_id: int,
        item_id: int | None = None,
        query: str | None = None,
        fallback_ids: list[int] | None = None,
        kinds: list[str] | None = None,
        on_date: str | None = None,
        at_time: str | None = None,
    ) -> tuple[Item | None, dict | None]:
        """Найти запись по id, по дате и времени, по описанию или по контексту диалога (последняя упомянутая)."""
        if item_id:
            item = self.repo.get_item(user_id, int(item_id))
            if not item:
                return None, {"status": "not_found", "message": f"Запись #{item_id} не найдена у этого пользователя."}
            return item, None
        pool = None
        if on_date or at_time:
            pool = self._items_at(user_id, on_date, at_time, kinds)
            if not pool:
                when = " ".join(filter(None, (on_date, at_time)))
                return None, {"status": "not_found", "message": f"На {when} записей нет."}
            if len(pool) == 1:
                return pool[0], None
            if not query:
                return None, self._ambiguous(pool)
        if query:
            results = self.search_service.search(user_id, query, limit=10, kinds=kinds, items=pool)
            if pool and not results:
                return None, self._ambiguous(pool)
            today = self.now().date()
            upcoming = [r for r in results if r[0].recurrence or r[0].start_date >= today]
            results = upcoming or results
            if not results:
                return None, {"status": "not_found", "message": f"Не нашел записей по запросу «{query}»."}
            full = [r for r in results if lexical_score(query, r[0].title) >= 0.999]
            top = results[0][1]
            close = full or [r for r in results if r[1] >= top - 0.08]
            if len(close) == 1:
                return close[0][0], None
            return None, self._ambiguous([i for i, _ in close])
        for rid in fallback_ids or []:
            item = self.repo.get_item(user_id, rid)
            if item:
                return item, None
        return None, _clarify(["item"], "Непонятно, о какой записи речь.")

    # ------------------------------------------------------------------ CRUD
    def create(
        self,
        user_id: int,
        kind: str,
        title: str | None = None,
        date: str | None = None,
        time: str | None = None,
        duration_min: int | None = None,
        recurrence_rule: dict | None = None,
        force: bool = False,
    ) -> dict:
        kind = (kind or "").strip().lower()
        if kind not in KINDS:
            return _clarify(["kind"], "Это задача, напоминание или встреча?")
        missing = []
        if not (title or "").strip():
            missing.append("title")
        if not date:
            missing.append("date")
        if kind in ("meeting", "reminder") and not time:
            missing.append("time")
        if missing:
            return _clarify(missing)
        try:
            item = Item(
                user_id=user_id,
                kind=kind,
                title=title.strip(),
                date=parse_date(date),
                time=parse_time(time) if time else None,
                duration_min=int(duration_min or self.meeting_default_minutes) if kind == "meeting" else None,
                recurrence=recurrence.normalize(recurrence_rule),
            )
        except ValueError as exc:
            return {"status": "error", "message": str(exc)}
        if not force:
            conflicts = self.find_conflicts(user_id, item)
            if conflicts:
                return {
                    "status": "conflict",
                    "conflicts": conflicts,
                    "message": "Встреча пересекается с другими встречами. Сообщите пользователю и спросите, "
                    "создать ли все равно (тогда вызовите снова с force=true) или выбрать другое время.",
                }
        self.repo.add_item(item)
        res = {"status": "ok", "action": "created", "item": self.view(item)}
        if item.start_dt() < self.now() and not item.recurrence:
            res["warning"] = "Дата и время уже в прошлом."
        return res

    def update(
        self,
        user_id: int,
        item_id: int | None = None,
        query: str | None = None,
        fallback_ids: list[int] | None = None,
        on_date: str | None = None,
        at_time: str | None = None,
        title: str | None = None,
        date: str | None = None,
        time: str | None = None,
        duration_min: int | None = None,
        kind: str | None = None,
        recurrence_rule: dict | None = None,
        clear_recurrence: bool = False,
        force: bool = False,
    ) -> dict:
        item, err = self.resolve(user_id, item_id, query, fallback_ids, on_date=on_date, at_time=at_time)
        if err:
            return err
        changes: dict = {}
        try:
            if title and title.strip() and title.strip() != item.title:
                changes["title"] = title.strip()
            if date:
                changes["date"] = parse_date(date)
            if time:
                changes["time"] = parse_time(time)
            if duration_min:
                changes["duration_min"] = int(duration_min)
            if kind:
                if kind not in KINDS:
                    raise ValidationError(f"Неизвестный тип записи: {kind}")
                changes["kind"] = kind
            if clear_recurrence:
                changes["recurrence"] = None
            elif recurrence_rule:
                changes["recurrence"] = recurrence.normalize(recurrence_rule)
        except ValueError as exc:
            return {"status": "error", "message": str(exc)}
        if not changes:
            return {"status": "need_clarification", "missing": ["changes"], "item": self.view(item),
                    "message": "Не понял, что изменить: название, дату или время? Уточните у пользователя."}
        updated = replace(item, **changes)
        if updated.kind == "meeting" and not updated.time:
            return _clarify(["time"], "Для встречи нужно время.")
        if updated.kind == "meeting" and updated.duration_min is None:
            changes["duration_min"] = updated.duration_min = self.meeting_default_minutes
        if not force and updated.kind == "meeting":
            conflicts = self.find_conflicts(user_id, updated, exclude_id=item.id)
            if conflicts:
                return {
                    "status": "conflict",
                    "item": self.view(item),
                    "conflicts": conflicts,
                    "message": "После переноса встреча пересечется с другими. Спросите пользователя, "
                    "переносить ли все равно (force=true).",
                }
        saved = self.repo.update_item(user_id, item.id, **changes)
        return {"status": "ok", "action": "updated", "before": self.view(item), "item": self.view(saved)}

    def delete(self, user_id: int, item_id: int | None = None, query: str | None = None,
               fallback_ids: list[int] | None = None, on_date: str | None = None,
               at_time: str | None = None) -> dict:
        item, err = self.resolve(user_id, item_id, query, fallback_ids, on_date=on_date, at_time=at_time)
        if err:
            return err
        self.repo.delete_item(user_id, item.id)
        return {"status": "ok", "action": "deleted", "item": self.view(item)}

    def complete(self, user_id: int, item_id: int | None = None, query: str | None = None,
                 fallback_ids: list[int] | None = None, on_date: str | None = None,
                 at_time: str | None = None, done: bool = True, occurrence_date: str | None = None) -> dict:
        item, err = self.resolve(user_id, item_id, query, fallback_ids, on_date=on_date, at_time=at_time)
        if err:
            return err
        if item.recurrence:
            try:
                d = date.fromisoformat(parse_date(occurrence_date)) if occurrence_date else \
                    recurrence.nearest_occurrence(item, self.now().date())
            except ValueError as exc:
                return {"status": "error", "message": str(exc)}
            if d is None:
                return {"status": "error", "message": "У повторяющейся записи нет экземпляров."}
            dates = set(item.done_dates)
            (dates.add if done else dates.discard)(d.isoformat())
            saved = self.repo.update_item(user_id, item.id, done_dates=sorted(dates))
            return {"status": "ok", "action": "completed" if done else "reopened",
                    "item": self.view(saved, d), "occurrence_date": d.isoformat()}
        saved = self.repo.update_item(user_id, item.id, status="done" if done else "open")
        return {"status": "ok", "action": "completed" if done else "reopened", "item": self.view(saved)}

    # ------------------------------------------------------------------ чтение
    def schedule(self, user_id: int, date_from: str, date_to: str | None = None,
                 kinds: list[str] | None = None, include_done: bool = True) -> list[dict]:
        start = date.fromisoformat(parse_date(date_from))
        end = date.fromisoformat(parse_date(date_to)) if date_to else start
        if end < start:
            start, end = end, start
        end = min(end, start + timedelta(days=MAX_PERIOD_DAYS))
        out = []
        for item in self.repo.list_items(user_id, date_to=end.isoformat(), kinds=kinds):
            for d in recurrence.occurrence_dates(item, start, end):
                v = self.view(item, d)
                if include_done or v["status"] != "done":
                    out.append(v)
        out.sort(key=lambda v: (v["date"], v["time"] or "", v["id"]))
        return out

    def upcoming(self, user_id: int, hours: int = 24, overdue_hours: int = 12) -> dict:
        """Ближайшие записи и наступившие (просроченные не более overdue_hours) напоминания и задачи."""
        now = self.now()
        occ = self.schedule(user_id, (now - timedelta(hours=overdue_hours)).date().isoformat(),
                            (now + timedelta(hours=hours)).date().isoformat(), include_done=False)
        due, upcoming = [], []
        for v in occ:
            at = datetime.combine(date.fromisoformat(v["date"]), time.fromisoformat(v["time"] or "00:00"))
            if v["time"] and now - timedelta(hours=overdue_hours) <= at <= now and v["kind"] in ("reminder", "task"):
                due.append(v)
            elif now < at <= now + timedelta(hours=hours) or (not v["time"] and v["date"] == now.date().isoformat()):
                upcoming.append(v)
        return {"now": now.isoformat(timespec="minutes"), "due": due, "upcoming": upcoming}

    def search(self, user_id: int, query: str, date_from: str | None = None, date_to: str | None = None,
               kinds: list[str] | None = None, limit: int = 10) -> dict:
        try:
            df = date.fromisoformat(parse_date(date_from)) if date_from else None
            dt = date.fromisoformat(parse_date(date_to)) if date_to else None
        except ValueError as exc:
            return {"status": "error", "message": str(exc)}
        found = self.search_service.search(user_id, query, limit=max(1, min(int(limit or 10), 30)),
                                           date_from=df, date_to=dt, kinds=kinds)
        results = []
        for item, score in found:
            v = self.view(item)
            v["score"] = round(score, 3)
            results.append(v)
        return {"status": "ok", "count": len(results), "results": results,
                "message": "" if results else "Ничего похожего не найдено."}


__all__ = ["Scheduler", "ValidationError", "parse_date", "parse_time", "KIND_RU"]

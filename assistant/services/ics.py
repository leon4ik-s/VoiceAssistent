"""Интеграция с внешними календарями через файлы .ics (iCalendar).

Экспорт: все записи пользователя -> VEVENT (повторы -> RRULE, напоминания -> VALARM).
Импорт: VEVENT из .ics -> встречи/задачи; повторный импорт того же файла обновляет записи по UID.
"""
from __future__ import annotations

import re
from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

from icalendar import Alarm, Calendar, Event

from ..storage.models import Item
from ..storage.repository import Repository
from . import recurrence as rec

SOURCE = "ics"
FREQ_TO_ICS = {"daily": "DAILY", "weekly": "WEEKLY", "monthly": "MONTHLY", "yearly": "YEARLY", "weekdays": "WEEKLY"}
ICS_TO_FREQ = {"DAILY": "daily", "WEEKLY": "weekly", "MONTHLY": "monthly", "YEARLY": "yearly"}
KIND_PREFIX = {"task": "Задача: ", "reminder": "Напоминание: ", "meeting": ""}


def export_ics(items: list[Item], tz_name: str, meeting_minutes: int = 60, host: str = "voice-assistant.local") -> bytes:
    tz = ZoneInfo(tz_name)
    cal = Calendar()
    cal.add("prodid", "-//Voice Assistant//RU")
    cal.add("version", "2.0")
    cal.add("x-wr-timezone", tz_name)
    stamp = datetime.now(tz)
    for it in items:
        ev = Event()
        ev.add("uid", it.external_id if it.source == SOURCE and it.external_id else f"item-{it.id}@{host}")
        ev.add("dtstamp", stamp)
        ev.add("summary", KIND_PREFIX.get(it.kind, "") + it.title)
        if it.time:
            start = datetime.combine(it.start_date, it.start_time).replace(tzinfo=tz)
            minutes = (it.duration_min or meeting_minutes) if it.kind == "meeting" else 15
            ev.add("dtstart", start)
            ev.add("dtend", start + timedelta(minutes=minutes))
        else:
            ev.add("dtstart", it.start_date)
            ev.add("dtend", it.start_date + timedelta(days=1))
        if it.recurrence:
            r = it.recurrence
            rr: dict = {"FREQ": FREQ_TO_ICS[r["freq"]]}
            if r.get("interval", 1) > 1:
                rr["INTERVAL"] = r["interval"]
            if r.get("weekdays"):
                rr["BYDAY"] = r["weekdays"]
            if r.get("count"):
                rr["COUNT"] = r["count"]
            elif r.get("until"):
                rr["UNTIL"] = datetime.combine(date.fromisoformat(r["until"]), time(23, 59, 59)).replace(tzinfo=tz)
            ev.add("rrule", rr)
        if it.status == "done":
            ev.add("description", "Выполнено")
        if it.kind == "reminder" and it.time:
            alarm = Alarm()
            alarm.add("action", "DISPLAY")
            alarm.add("description", it.title)
            alarm.add("trigger", timedelta(minutes=0))
            ev.add_component(alarm)
        cal.add_component(ev)
    return cal.to_ical()


def _to_local(value, tz: ZoneInfo) -> tuple[date, str | None]:
    if isinstance(value, datetime):
        if value.tzinfo is not None:
            value = value.astimezone(tz)
        return value.date(), value.strftime("%H:%M")
    return value, None


def _parse_rrule(rrule) -> dict | None:
    if not rrule:
        return None
    freq = ICS_TO_FREQ.get(str((rrule.get("FREQ") or [""])[0]).upper())
    if not freq:
        return None
    out: dict = {"freq": freq}
    if rrule.get("INTERVAL"):
        out["interval"] = int(rrule["INTERVAL"][0])
    if rrule.get("BYDAY"):
        days = [str(d)[-2:].upper() for d in rrule["BYDAY"]]
        out["weekdays"] = days
    if rrule.get("COUNT"):
        out["count"] = int(rrule["COUNT"][0])
    if rrule.get("UNTIL"):
        u = rrule["UNTIL"][0]
        out["until"] = (u.date() if isinstance(u, datetime) else u).isoformat()
    return rec.normalize(out)


def import_ics(repo: Repository, user_id: int, data: bytes, tz_name: str) -> dict:
    tz = ZoneInfo(tz_name)
    cal = Calendar.from_ical(data)
    created = updated = skipped = 0
    for ev in cal.walk("VEVENT"):
        try:
            uid = str(ev.get("uid") or "").strip()
            title = str(ev.get("summary") or "").strip() or "Событие из календаря"
            start = ev.get("dtstart")
            if start is None:
                raise ValueError("нет DTSTART")
            d, t = _to_local(start.dt, tz)
            duration = None
            if t and ev.get("dtend") is not None:
                end = ev.get("dtend").dt
                if isinstance(end, datetime) and isinstance(start.dt, datetime):
                    duration = max(1, int((end - start.dt).total_seconds() // 60))
            kind = "meeting" if t else "task"
            for k, prefix in KIND_PREFIX.items():
                if prefix and title.startswith(prefix):
                    kind, title = k, title[len(prefix):].strip() or title
            item = Item(user_id=user_id, kind=kind, title=title, date=d.isoformat(), time=t,
                        duration_min=(duration or 60) if kind == "meeting" else None,
                        recurrence=_parse_rrule(ev.get("rrule")), source=SOURCE,
                        external_id=uid or None)
        except Exception:
            skipped += 1
            continue
        existing = repo.find_by_external(user_id, SOURCE, uid) if uid else None
        own = re.fullmatch(r"item-(\d+)@.+", uid)
        if existing is None and own:  # файл, выгруженный из этого же приложения
            existing = repo.get_item(user_id, int(own.group(1)))
        if existing:
            repo.update_item(user_id, existing.id, kind=item.kind, title=item.title, date=item.date,
                             time=item.time, duration_min=item.duration_min, recurrence=item.recurrence)
            updated += 1
        else:
            repo.add_item(item)
            created += 1
    return {"created": created, "updated": updated, "skipped": skipped}

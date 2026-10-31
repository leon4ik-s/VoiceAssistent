"""Экспорт и импорт .ics."""
from icalendar import Calendar

from assistant.services import ics


def test_export_contains_events_rrule_and_alarm(ctx, user):
    s = ctx.scheduler
    s.create(user, "meeting", "Планерка", "2026-11-02", "10:00", duration_min=30, recurrence_rule={"freq": "weekdays"})
    s.create(user, "reminder", "Позвонить маме", "2026-11-03", "09:00")
    s.create(user, "task", "Отчет", "2026-11-04")
    data = ics.export_ics(ctx.repo.list_items(user), "Europe/Moscow")
    cal = Calendar.from_ical(data)
    events = list(cal.walk("VEVENT"))
    assert len(events) == 3
    by_title = {str(e["summary"]): e for e in events}
    plan = by_title["Планерка"]
    assert plan["rrule"]["FREQ"] == ["WEEKLY"] and len(plan["rrule"]["BYDAY"]) == 5
    assert (plan["dtend"].dt - plan["dtstart"].dt).seconds == 30 * 60
    assert list(by_title["Напоминание: Позвонить маме"].walk("VALARM"))
    assert "Задача: Отчет" in by_title


def test_import_roundtrip_and_idempotent(ctx, user, other_user):
    s = ctx.scheduler
    s.create(user, "meeting", "Планерка", "2026-11-02", "10:00", recurrence_rule={"freq": "weekly", "count": 4})
    s.create(user, "reminder", "Позвонить маме", "2026-11-03", "09:00")
    s.create(user, "task", "Отчет", "2026-11-04")
    data = ics.export_ics(ctx.repo.list_items(user), "Europe/Moscow")

    r = ics.import_ics(ctx.repo, other_user, data, "Europe/Moscow")
    assert r == {"created": 3, "updated": 0, "skipped": 0}
    again = ics.import_ics(ctx.repo, other_user, data, "Europe/Moscow")
    assert again["created"] == 0 and again["updated"] == 3
    items = {i.title: i for i in ctx.repo.list_items(other_user)}
    assert items["Планерка"].kind == "meeting" and items["Планерка"].recurrence["count"] == 4
    assert items["Позвонить маме"].kind == "reminder" and items["Позвонить маме"].time == "09:00"
    assert items["Отчет"].kind == "task" and items["Отчет"].time is None
    # свой же файл, загруженный обратно, обновляет записи, а не дублирует
    assert ics.import_ics(ctx.repo, user, data, "Europe/Moscow")["created"] == 0
    assert ctx.repo.count_items(user) == 3


def test_import_external_utc_event(ctx, user):
    data = """BEGIN:VCALENDAR
VERSION:2.0
PRODID:-//Test//EN
BEGIN:VEVENT
UID:abc-123@google.com
DTSTAMP:20261030T070000Z
DTSTART:20261105T070000Z
DTEND:20261105T083000Z
SUMMARY:Встреча с подрядчиком
END:VEVENT
END:VCALENDAR
""".encode("utf-8")
    assert ics.import_ics(ctx.repo, user, data, "Europe/Moscow")["created"] == 1
    item = ctx.repo.list_items(user)[0]
    assert (item.date, item.time, item.duration_min, item.kind) == ("2026-11-05", "10:00", 90, "meeting")

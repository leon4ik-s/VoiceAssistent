"""Бизнес-логика: создание, правка, удаление, выполнение, повторы, конфликты, период, пользователи."""
from assistant.providers import build_context

from conftest import NOW


def test_create_all_kinds_and_persist(ctx, settings, providers, user):
    s = ctx.scheduler
    r1 = s.create(user, "task", "Подготовить отчет", "2026-10-31")
    r2 = s.create(user, "reminder", "Позвонить маме", "2026-10-31", "09:00")
    r3 = s.create(user, "meeting", "Встреча с Иваном", "2026-10-31", "15:00")
    assert [r["status"] for r in (r1, r2, r3)] == ["ok", "ok", "ok"]
    assert r3["item"]["duration_min"] == 60 and r3["item"]["end_time"] == "16:00"
    ctx.db.close()
    # данные живут после «перезапуска»
    ctx2 = build_context(settings, providers, now_fn=lambda: NOW)
    items = ctx2.scheduler.schedule(user, "2026-10-31", "2026-10-31")
    assert [i["title"] for i in items] == ["Подготовить отчет", "Позвонить маме", "Встреча с Иваном"]
    ctx2.db.close()


def test_missing_data_requires_clarification(ctx, user):
    s = ctx.scheduler
    r = s.create(user, "meeting", "Встреча с Иваном", "2026-10-31")
    assert r["status"] == "need_clarification" and r["missing"] == ["time"]
    r = s.create(user, "reminder", "Позвонить", None, None)
    assert r["status"] == "need_clarification" and set(r["missing"]) == {"date", "time"}
    r = s.create(user, "", "Что-то", "2026-10-31")
    assert r["status"] == "need_clarification" and r["missing"] == ["kind"]
    assert ctx.repo.count_items(user) == 0


def test_invalid_values(ctx, user):
    assert ctx.scheduler.create(user, "task", "X", "31.10.2026")["status"] == "error"
    assert ctx.scheduler.create(user, "reminder", "X", "2026-10-31", "25:00")["status"] == "error"


def test_update_delete_complete(ctx, user):
    s = ctx.scheduler
    item_id = s.create(user, "task", "Отчет", "2026-10-31", "12:00")["item"]["id"]
    r = s.update(user, item_id, title="Квартальный отчет", date="2026-11-02", time="14:30")
    assert r["status"] == "ok"
    assert (r["item"]["title"], r["item"]["date"], r["item"]["time"]) == ("Квартальный отчет", "2026-11-02", "14:30")
    assert s.update(user, item_id)["status"] == "need_clarification"
    assert s.complete(user, item_id)["item"]["status"] == "done"
    assert s.complete(user, item_id, done=False)["item"]["status"] == "open"
    assert s.delete(user, item_id)["status"] == "ok"
    assert ctx.repo.get_item(user, item_id) is None
    assert s.delete(user, item_id)["status"] == "not_found"


def test_recurrence_expansion(ctx, user):
    s = ctx.scheduler
    s.create(user, "task", "Зарядка", "2026-11-02", "07:00", recurrence_rule={"freq": "daily", "count": 5})
    s.create(user, "meeting", "Планерка", "2026-11-02", "10:00", recurrence_rule={"freq": "weekdays"})
    s.create(user, "reminder", "Оплатить интернет", "2026-11-05", "12:00",
             recurrence_rule={"freq": "monthly", "until": "2027-01-31"})
    s.create(user, "meeting", "Спортзал", "2026-11-03", "19:00",
             recurrence_rule={"freq": "weekly", "weekdays": ["TU", "TH"]})
    week = s.schedule(user, "2026-11-02", "2026-11-08")
    titles = [i["title"] for i in week]
    assert titles.count("Зарядка") == 5
    assert titles.count("Планерка") == 5          # пн-пт
    assert titles.count("Спортзал") == 2          # вт и чт
    assert titles.count("Оплатить интернет") == 1
    pay = [i for i in s.schedule(user, "2026-11-01", "2027-03-31") if i["title"] == "Оплатить интернет"]
    assert [i["date"] for i in pay] == ["2026-11-05", "2026-12-05", "2027-01-05"]
    assert all(i["recurrence_ru"] for i in week)


def test_complete_single_occurrence_of_recurring(ctx, user):
    s = ctx.scheduler
    item_id = s.create(user, "task", "Полить цветы", "2026-10-30", "09:00",
                       recurrence_rule={"freq": "daily"})["item"]["id"]
    r = s.complete(user, item_id, occurrence_date="2026-10-31")
    assert r["status"] == "ok" and r["occurrence_date"] == "2026-10-31"
    days = {i["date"]: i["status"] for i in s.schedule(user, "2026-10-30", "2026-11-01")}
    assert days == {"2026-10-30": "open", "2026-10-31": "done", "2026-11-01": "open"}


def test_meeting_conflicts(ctx, user):
    s = ctx.scheduler
    first = s.create(user, "meeting", "Встреча с Иваном", "2026-11-02", "10:00")["item"]["id"]
    r = s.create(user, "meeting", "Созвон с Марией", "2026-11-02", "10:30", duration_min=30)
    assert r["status"] == "conflict" and r["conflicts"][0]["id"] == first
    assert ctx.repo.count_items(user) == 1
    assert s.create(user, "meeting", "Созвон с Марией", "2026-11-02", "10:30", force=True)["status"] == "ok"
    # стык встреч не пересечение
    assert s.create(user, "meeting", "Обед", "2026-11-02", "11:30")["status"] == "ok"
    # задачи с конфликтами не проверяются
    assert s.create(user, "task", "Отчет", "2026-11-02", "10:00")["status"] == "ok"


def test_conflict_with_recurring_and_on_reschedule(ctx, user):
    s = ctx.scheduler
    s.create(user, "meeting", "Планерка", "2026-11-02", "09:00", duration_min=30, recurrence_rule={"freq": "weekdays"})
    r = s.create(user, "meeting", "Демо", "2026-11-12", "09:15")
    assert r["status"] == "conflict" and r["conflicts"][0]["date"] == "2026-11-12"
    assert s.create(user, "meeting", "Демо", "2026-11-14", "09:15")["status"] == "ok"  # суббота
    other = s.create(user, "meeting", "Созвон", "2026-11-12", "12:00")["item"]["id"]
    assert s.update(user, other, time="09:00")["status"] == "conflict"
    assert ctx.repo.get_item(user, other).time == "12:00"
    assert s.update(user, other, time="09:00", force=True)["status"] == "ok"


def test_schedule_period_and_filter(ctx, user):
    s = ctx.scheduler
    s.create(user, "task", "A", "2026-11-01")
    s.create(user, "meeting", "B", "2026-11-03", "10:00")
    s.create(user, "reminder", "C", "2026-11-07", "10:00")
    s.create(user, "task", "D", "2026-11-08")
    assert [i["title"] for i in s.schedule(user, "2026-11-01", "2026-11-07")] == ["A", "B", "C"]
    assert [i["title"] for i in s.schedule(user, "2026-11-01", "2026-11-07", ["meeting"])] == ["B"]
    assert [i["title"] for i in s.schedule(user, "2026-11-08")] == ["D"]


def test_users_are_isolated(ctx, user, other_user):
    s = ctx.scheduler
    item_id = s.create(user, "meeting", "Встреча с Иваном", "2026-11-02", "10:00")["item"]["id"]
    assert s.schedule(other_user, "2026-11-01", "2026-11-30") == []
    assert s.update(other_user, item_id, title="Взлом")["status"] == "not_found"
    assert s.delete(other_user, item_id)["status"] == "not_found"
    assert s.search(other_user, "Иван")["count"] == 0
    # у второго пользователя в то же время конфликта нет
    assert s.create(other_user, "meeting", "Своя встреча", "2026-11-02", "10:00")["status"] == "ok"
    assert ctx.repo.get_item(user, item_id).title == "Встреча с Иваном"


def test_resolve_by_query_unique_and_ambiguous(ctx, user):
    s = ctx.scheduler
    s.create(user, "meeting", "Встреча с Иваном по договору", "2026-11-02", "10:00")
    s.create(user, "reminder", "Позвонить Ивану", "2026-11-03", "12:00")
    s.create(user, "task", "Купить корм коту", "2026-11-04")
    r = s.delete(user, query="Иван")
    assert r["status"] == "ambiguous" and len(r["candidates"]) == 2
    assert ctx.repo.count_items(user) == 3
    r = s.delete(user, query="корм коту")
    assert r["status"] == "ok" and r["item"]["title"] == "Купить корм коту"


def test_upcoming_and_due(ctx, user):
    s = ctx.scheduler
    s.create(user, "reminder", "Выпить таблетку", "2026-10-30", "09:30")   # уже наступило
    s.create(user, "meeting", "Встреча", "2026-10-30", "15:00")            # впереди
    s.create(user, "task", "Через неделю", "2026-11-06", "10:00")          # вне окна
    up = s.upcoming(user, hours=24)
    assert [i["title"] for i in up["due"]] == ["Выпить таблетку"]
    assert [i["title"] for i in up["upcoming"]] == ["Встреча"]


def test_resolve_by_date_and_time(ctx, user):
    s = ctx.scheduler
    evening = s.create(user, "meeting", "Встреча", "2026-11-02", "21:00")["item"]
    morning = s.create(user, "meeting", "Встреча с Иваном", "2026-11-02", "10:00")["item"]
    s.create(user, "meeting", "Встреча с Анной", "2026-11-03", "21:00")
    assert s.resolve(user, on_date="2026-11-02", at_time="21:00")[0].id == evening["id"]
    assert s.resolve(user, on_date="2026-11-02", query="Иван")[0].id == morning["id"]
    assert s.resolve(user, query="встреча", at_time="10:00")[0].id == morning["id"]

    _, err = s.resolve(user, on_date="2026-11-02")
    assert err["status"] == "ambiguous" and len(err["candidates"]) == 2
    _, err = s.resolve(user, on_date="2026-11-02", query="совершенно другое")
    assert err["status"] == "ambiguous"  # описание не помогло, но дата уже сузила выбор
    _, err = s.resolve(user, on_date="2026-11-05", at_time="21:00")
    assert err["status"] == "not_found"


def test_resolve_by_date_finds_recurring_occurrence(ctx, user):
    s = ctx.scheduler
    item = s.create(user, "meeting", "Планерка", "2026-11-02", "10:00", recurrence_rule={"freq": "daily"})["item"]
    assert s.resolve(user, on_date="2026-11-10", at_time="10:00")[0].id == item["id"]

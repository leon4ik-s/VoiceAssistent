"""Диалоговый слой с фейковой LLM: инструменты, уточнения, контекст «перенеси ее», конфликты."""
import json

import pytest

from assistant.config import ConfigError
from assistant.nlu.agent import Assistant

from conftest import ScriptedLLM


def tool_results(llm: ScriptedLLM) -> list[dict]:
    last = llm.requests[-1]
    return [json.loads(m["content"]) for m in last if m["role"] == "tool"]


def test_create_via_tools_and_history(ctx, llm_holder, user):
    llm_holder["llm"] = llm = ScriptedLLM([
        [("create_item", {"kind": "meeting", "title": "Встреча с Иваном", "date": "2026-10-31", "time": "15:00"})],
        "Записал встречу с Иваном на завтра в 15:00.",
    ])
    res = ctx.assistant.handle(user, "Поставь встречу с Иваном завтра в три часа дня")
    assert res.reply.startswith("Записал") and res.changed
    assert tool_results(llm)[0]["status"] == "ok"
    items = ctx.scheduler.schedule(user, "2026-10-31")
    assert [(i["title"], i["time"]) for i in items] == [("Встреча с Иваном", "15:00")]
    # в системный промпт передается текущая дата и часовой пояс
    system = llm.requests[0][0]["content"]
    assert "2026-10-30 10:00" in system and "Europe/Moscow" in system
    hist = ctx.repo.recent_messages(user, 10)
    assert [m["role"] for m in hist] == ["user", "tool_log", "assistant"]


def test_move_it_to_tomorrow_uses_context(ctx, llm_holder, user):
    llm_holder["llm"] = ScriptedLLM([
        [("create_item", {"kind": "meeting", "title": "Созвон с Марией", "date": "2026-10-30", "time": "16:00"})],
        "Готово.",
    ])
    ctx.assistant.handle(user, "Созвон с Марией сегодня в 16")
    created = ctx.repo.list_items(user)[0]

    # Модель не передает item_id - запись берется из контекста диалога
    llm_holder["llm"] = llm = ScriptedLLM([[("update_item", {"date": "2026-10-31"})], "Перенес на завтра."])
    res = ctx.assistant.handle(user, "Перенеси ее на завтра")
    assert res.changed
    moved = ctx.repo.get_item(user, created.id)
    assert (moved.date, moved.time) == ("2026-10-31", "16:00")
    # и сама запись видна модели в контексте
    assert f"#{created.id}" in llm.requests[0][0]["content"]
    # предыдущие реплики тоже переданы
    assert any(m["role"] == "user" and "Марией" in m["content"] for m in llm.requests[0])


def test_clarification_when_time_missing(ctx, llm_holder, user):
    llm_holder["llm"] = llm = ScriptedLLM([
        [("create_item", {"kind": "meeting", "title": "Встреча с Олегом", "date": "2026-11-02"})],
        "Во сколько встреча с Олегом в понедельник?",
    ])
    res = ctx.assistant.handle(user, "Запиши встречу с Олегом в понедельник")
    assert res.reply.startswith("Во сколько") and not res.changed
    assert tool_results(llm)[0]["status"] == "need_clarification"
    assert ctx.repo.count_items(user) == 0

    llm_holder["llm"] = ScriptedLLM([
        [("create_item", {"kind": "meeting", "title": "Встреча с Олегом", "date": "2026-11-02", "time": "11:00"})],
        "Записал.",
    ])
    ctx.assistant.handle(user, "В 11")
    assert ctx.repo.count_items(user) == 1


def test_conflict_then_confirm(ctx, llm_holder, user):
    ctx.scheduler.create(user, "meeting", "Планерка", "2026-11-02", "10:00")
    args = {"kind": "meeting", "title": "Демо", "date": "2026-11-02", "time": "10:30"}
    llm_holder["llm"] = llm = ScriptedLLM([[("create_item", args)], "Пересекается с планеркой. Все равно поставить?"])
    ctx.assistant.handle(user, "Демо в понедельник в 10:30")
    assert tool_results(llm)[0]["status"] == "conflict"
    assert ctx.repo.count_items(user) == 1
    # в истории сохранен журнал вызовов, модель видит его на следующем шаге
    llm_holder["llm"] = llm = ScriptedLLM([[("create_item", {**args, "force": True})], "Поставил."])
    ctx.assistant.handle(user, "Да, ставь")
    system = llm.requests[0][0]
    assert system["role"] == "system" and "Журнал действий" in system["content"] and "conflict" in system["content"]
    assert not any("conflict" in m["content"] for m in llm.requests[0][1:])
    assert ctx.repo.count_items(user) == 2


def test_ambiguous_delete_asks(ctx, llm_holder, user):
    ctx.scheduler.create(user, "meeting", "Встреча с Иваном", "2026-11-02", "10:00")
    ctx.scheduler.create(user, "reminder", "Позвонить Ивану", "2026-11-03", "12:00")
    llm_holder["llm"] = llm = ScriptedLLM([[("delete_item", {"query": "Иван"})], "Какую удалить: встречу или звонок?"])
    ctx.assistant.handle(user, "Удали про Ивана")
    r = tool_results(llm)[0]
    assert r["status"] == "ambiguous" and len(r["candidates"]) == 2
    assert ctx.repo.count_items(user) == 2


def test_complete_and_search_via_tools(ctx, llm_holder, user):
    item = ctx.scheduler.create(user, "task", "Отправить договор Ивану", "2026-10-30")["item"]
    llm_holder["llm"] = ScriptedLLM([[("complete_item", {"item_id": item["id"]})], "Отметил."])
    ctx.assistant.handle(user, "Договор Ивану отправил")
    assert ctx.repo.get_item(user, item["id"]).status == "done"
    llm_holder["llm"] = llm = ScriptedLLM([[("search_items", {"query": "договор с Иваном"})], "Было 30 октября."])
    ctx.assistant.handle(user, "Что у меня было про договор с Иваном?")
    r = tool_results(llm)[0]
    assert r["count"] == 1 and r["results"][0]["id"] == item["id"]


def test_tool_loop_is_bounded(ctx, llm_holder, user):
    llm_holder["llm"] = ScriptedLLM([[("list_schedule", {"date_from": "2026-10-30", "date_to": "2026-10-30"})]] * 10)
    res = ctx.assistant.handle(user, "Покажи расписание")
    assert "Не получилось" in res.reply


def test_no_api_key_gives_config_error(ctx, no_key_providers, user):
    a = Assistant(ctx.repo, ctx.scheduler, no_key_providers.llm)
    with pytest.raises(ConfigError):
        a.handle(user, "Привет")
    assert ctx.repo.recent_messages(user, 10) == []


FAKE_CALL = 'Служебно, ранее вызванные инструменты: delete_item({"query": "встреча завтра в 21:00"}) -> success\n\n Ассистент: Встреча удалена.'


def test_tool_call_written_as_text_is_retried(ctx, llm_holder, user):
    item = ctx.scheduler.create(user, "meeting", "Встреча", "2026-10-31", "21:00")["item"]
    llm_holder["llm"] = llm = ScriptedLLM([FAKE_CALL, [("delete_item", {"item_id": item["id"]})], "Удалил."])
    res = ctx.assistant.handle(user, "Удали которая завтра в 21:00")
    assert res.reply == "Удалил." and res.changed
    assert ctx.repo.count_items(user) == 0
    assert "НЕ выполнено" in llm.requests[1][-1]["content"]
    assert FAKE_CALL not in str(llm.requests[1])  # выдуманный вызов не остается в диалоге


def test_action_not_reported_when_model_only_writes_it(ctx, llm_holder, user):
    ctx.scheduler.create(user, "meeting", "Встреча", "2026-10-31", "21:00")
    llm_holder["llm"] = ScriptedLLM([FAKE_CALL, FAKE_CALL])
    res = ctx.assistant.handle(user, "Удали которая завтра в 21:00")
    assert res.reply == "Не получилось выполнить запрос. Попробуйте сформулировать иначе."
    assert not res.changed and ctx.repo.count_items(user) == 1


def test_service_text_removed_after_real_action(ctx, llm_holder, user):
    item = ctx.scheduler.create(user, "meeting", "Встреча", "2026-10-31", "21:00")["item"]
    llm_holder["llm"] = ScriptedLLM([[("delete_item", {"item_id": item["id"]})], FAKE_CALL])
    res = ctx.assistant.handle(user, "Удали встречу")
    assert res.reply == "Встреча удалена." and res.changed


def test_user_picks_from_ambiguous_variants_by_id(ctx, llm_holder, user):
    first = ctx.scheduler.create(user, "meeting", "Встреча", "2026-10-31", "21:00")["item"]
    ctx.scheduler.create(user, "meeting", "Встреча с Алексеем", "2027-08-14", "17:00")
    llm_holder["llm"] = ScriptedLLM([[("delete_item", {"query": "встреча"})], "Какую удалить?"])
    ctx.assistant.handle(user, "Удали встречу")
    assert ctx.repo.get_recent_ids(user) == []  # варианты не считаются упомянутой записью

    llm_holder["llm"] = llm = ScriptedLLM([[("delete_item", {"item_id": first["id"]})], "Удалил."])
    ctx.assistant.handle(user, "Которая завтра в 21:00")
    assert f"#{first['id']} Встреча 2026-10-31 21:00" in llm.requests[0][0]["content"]
    assert [i.title for i in ctx.repo.list_items(user)] == ["Встреча с Алексеем"]


def test_delete_by_date_and_time_via_tool(ctx, llm_holder, user):
    ctx.scheduler.create(user, "meeting", "Встреча", "2026-10-31", "21:00")
    ctx.scheduler.create(user, "meeting", "Встреча с Алексеем", "2027-08-14", "17:00")
    llm_holder["llm"] = ScriptedLLM([[("delete_item", {"on_date": "завтра", "at_time": "21"})], "Удалил."])
    ctx.assistant.handle(user, "Удали которая завтра в 9 вечера")
    assert [i.title for i in ctx.repo.list_items(user)] == ["Встреча с Алексеем"]

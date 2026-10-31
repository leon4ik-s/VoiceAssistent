"""Описание инструментов (function calling) и их выполнение через бизнес-логику."""
from __future__ import annotations

import re
from datetime import date, timedelta
from typing import Any

from ..services.scheduler import Scheduler
from .llm import parse_json_lenient

KIND_ENUM = ["task", "reminder", "meeting"]

RECURRENCE_SCHEMA = {
    "type": "object",
    "description": "Правило повтора. Не передавать, если событие разовое.",
    "properties": {
        "freq": {"type": "string", "enum": ["daily", "weekly", "monthly", "yearly", "weekdays"],
                 "description": "weekdays = по будням (пн-пт)"},
        "interval": {"type": "integer", "description": "Каждые N периодов, по умолчанию 1"},
        "weekdays": {"type": "array", "items": {"type": "string", "enum": ["MO", "TU", "WE", "TH", "FR", "SA", "SU"]},
                     "description": "Дни недели для weekly"},
        "until": {"type": "string", "description": "Дата окончания YYYY-MM-DD"},
        "count": {"type": "integer", "description": "Количество повторов"},
    },
    "required": ["freq"],
}

REF_PROPS = {
    "item_id": {"type": "integer", "description": "id записи, если известен (из контекста или результатов поиска)"},
    "query": {"type": "string", "description": "Описание записи словами, если id неизвестен"},
    "on_date": {"type": "string", "description": "Текущая дата записи YYYY-MM-DD, если запись называют по дате"},
    "at_time": {"type": "string", "description": "Текущее время записи HH:MM, если запись называют по времени"},
}


def _fn(name: str, description: str, properties: dict, required: list[str] | None = None) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": properties, "required": required or []},
        },
    }


TOOLS: list[dict] = [
    _fn(
        "create_item",
        "Создать задачу, напоминание или встречу. Вызывай только когда известны название и дата "
        "(для встречи и напоминания еще и время). Не выдумывай недостающие данные.",
        {
            "kind": {"type": "string", "enum": KIND_ENUM},
            "title": {"type": "string", "description": "Короткое название без даты и времени"},
            "date": {"type": "string", "description": "Дата YYYY-MM-DD"},
            "time": {"type": "string", "description": "Время HH:MM (24ч)"},
            "duration_min": {"type": "integer", "description": "Длительность встречи в минутах"},
            "recurrence": RECURRENCE_SCHEMA,
            "force": {"type": "boolean", "description": "true - создать несмотря на пересечение встреч"},
        },
        ["kind", "title"],
    ),
    _fn(
        "update_item",
        "Изменить запись: название, дату, время, длительность, повтор. Если пользователь говорит "
        "«ее», «его», «эту» - это последняя упомянутая запись из контекста.",
        {
            **REF_PROPS,
            "title": {"type": "string"},
            "date": {"type": "string", "description": "Новая дата YYYY-MM-DD"},
            "time": {"type": "string", "description": "Новое время HH:MM"},
            "duration_min": {"type": "integer"},
            "kind": {"type": "string", "enum": KIND_ENUM},
            "recurrence": RECURRENCE_SCHEMA,
            "clear_recurrence": {"type": "boolean", "description": "Сделать событие разовым"},
            "force": {"type": "boolean", "description": "true - перенести несмотря на пересечение"},
        },
    ),
    _fn("delete_item", "Удалить запись (для повторяющейся - всю серию).", dict(REF_PROPS)),
    _fn(
        "complete_item",
        "Отметить задачу выполненной (done=false - вернуть в работу). Для повторяющейся отмечается один экземпляр.",
        {
            **REF_PROPS,
            "done": {"type": "boolean", "description": "по умолчанию true"},
            "occurrence_date": {"type": "string", "description": "Дата экземпляра повторяющейся задачи"},
        },
    ),
    _fn(
        "list_schedule",
        "Показать расписание за период (включительно).",
        {
            "date_from": {"type": "string", "description": "YYYY-MM-DD"},
            "date_to": {"type": "string", "description": "YYYY-MM-DD"},
            "kind": {"type": "string", "enum": KIND_ENUM, "description": "Фильтр по типу"},
            "include_done": {"type": "boolean"},
        },
        ["date_from", "date_to"],
    ),
    _fn(
        "search_items",
        "Поиск записей по смыслу, в том числе в истории прошлых запросов. Используй для вопросов "
        "«что у меня было про...», «когда была...», и чтобы найти id записи перед правкой.",
        {
            "query": {"type": "string", "description": "Смысл запроса своими словами, без слов вроде «когда была»"},
            "date_from": {"type": "string"},
            "date_to": {"type": "string"},
            "kind": {"type": "string", "enum": KIND_ENUM},
            "limit": {"type": "integer", "description": "по умолчанию 10"},
        },
        ["query"],
    ),
]


def _kinds(args: dict) -> list[str] | None:
    k = args.get("kind")
    return [k] if k in KIND_ENUM else None


TOOL_INDEX = {t["function"]["name"]: t["function"] for t in TOOLS}
TRUE_WORDS = {"true", "1", "yes", "да", "y"}
FALSE_WORDS = {"false", "0", "no", "нет", "n"}
RELATIVE_DAYS = {"сегодня": 0, "today": 0, "завтра": 1, "tomorrow": 1, "послезавтра": 2, "вчера": -1}
DATE_FIELDS = {"date", "date_from", "date_to", "occurrence_date", "on_date"}
TIME_FIELDS = {"time", "at_time"}


class ArgError(ValueError):
    pass


def normalize_date(value: Any, today: date) -> Any:
    """Подстраховка для локальных моделей: «завтра», 31.10.2026, 31.10 -> YYYY-MM-DD."""
    v = str(value).strip().lower()
    if v in RELATIVE_DAYS:
        return (today + timedelta(days=RELATIVE_DAYS[v])).isoformat()
    m = re.fullmatch(r"(\d{1,2})[./](\d{1,2})(?:[./](\d{2,4}))?", v)
    if m:
        d, mo, y = int(m.group(1)), int(m.group(2)), m.group(3)
        year = int(y) + (2000 if y and len(y) == 2 else 0) if y else today.year
        try:
            return date(year, mo, d).isoformat()
        except ValueError:
            return value
    return value


def normalize_time(value: Any) -> Any:
    v = str(value).strip().lower().replace(" ", "")
    m = re.fullmatch(r"(\d{1,2})(?:ч|h|:00)?", v)
    if m and int(m.group(1)) < 24:
        return f"{int(m.group(1)):02d}:00"
    m = re.fullmatch(r"(\d{1,2})[:.](\d{2})(?::\d{2})?", v)
    if m:
        return f"{int(m.group(1)):02d}:{m.group(2)}"
    return value


def _coerce(name: str, spec: dict, value: Any) -> Any:
    t = spec.get("type")
    if t == "integer":
        if isinstance(value, bool):
            raise ArgError(f"{name}: нужно целое число")
        try:
            return int(float(str(value).strip()))
        except ValueError:
            raise ArgError(f"{name}: нужно целое число, получено {value!r}")
    if t == "boolean":
        if isinstance(value, bool):
            return value
        v = str(value).strip().lower()
        if v in TRUE_WORDS:
            return True
        if v in FALSE_WORDS:
            return False
        raise ArgError(f"{name}: нужно true или false")
    if t == "string":
        v = str(value).strip()
        if "enum" in spec:
            v = v.lower()
            if v not in spec["enum"]:
                raise ArgError(f"{name}: допустимо {', '.join(spec['enum'])}, получено {value!r}")
        return v
    if t == "array":
        if isinstance(value, str):
            value = [x.strip() for x in value.split(",") if x.strip()]
        if not isinstance(value, list):
            raise ArgError(f"{name}: нужен список")
        return value
    if t == "object":
        if isinstance(value, str):
            if value.strip().lower() in ("daily", "weekly", "monthly", "yearly", "weekdays"):
                return {"freq": value.strip().lower()}
            try:
                value = parse_json_lenient(value)
            except ValueError:
                raise ArgError(f"{name}: нужен JSON-объект")
        if not isinstance(value, dict):
            raise ArgError(f"{name}: нужен JSON-объект")
        return value
    return value


def validate_args(name: str, args: dict, today: date) -> dict:
    """Строгая проверка аргументов по схеме инструмента: лишнее отбросить, типы привести, обязательные проверить."""
    fn = TOOL_INDEX.get(name)
    if fn is None:
        raise ArgError(f"Неизвестный инструмент {name}. Доступны: {', '.join(TOOL_INDEX)}")
    props = fn["parameters"]["properties"]
    clean: dict = {}
    for key, value in (args or {}).items():
        if key not in props or value is None or (isinstance(value, str) and not value.strip()):
            continue
        v = _coerce(key, props[key], value)
        if key in DATE_FIELDS:
            v = normalize_date(v, today)
        elif key in TIME_FIELDS:
            v = normalize_time(v)
        clean[key] = v
    missing = [r for r in fn["parameters"].get("required", []) if r not in clean]
    if missing:
        raise ArgError(f"Не хватает параметров: {', '.join(missing)}. Если пользователь их не назвал, спроси его.")
    return clean


def execute(scheduler: Scheduler, user_id: int, name: str, args: dict, recent_ids: list[int]) -> dict[str, Any]:
    """Выполнить инструмент. Ошибки возвращаются модели как данные, чтобы она могла исправиться или переспросить."""
    try:
        a = validate_args(name, args, scheduler.now().date())
    except ArgError as exc:
        return {"status": "error", "message": f"{exc} Исправь аргументы и вызови инструмент снова."}
    try:
        if name == "create_item":
            return scheduler.create(user_id, a.get("kind", ""), a.get("title"), a.get("date"), a.get("time"),
                                    a.get("duration_min"), a.get("recurrence"), bool(a.get("force")))
        if name == "update_item":
            return scheduler.update(user_id, a.get("item_id"), a.get("query"), recent_ids,
                                    on_date=a.get("on_date"), at_time=a.get("at_time"), title=a.get("title"), date=a.get("date"), time=a.get("time"),
                                    duration_min=a.get("duration_min"), kind=a.get("kind"),
                                    recurrence_rule=a.get("recurrence"),
                                    clear_recurrence=bool(a.get("clear_recurrence")), force=bool(a.get("force")))
        if name == "delete_item":
            return scheduler.delete(user_id, a.get("item_id"), a.get("query"), recent_ids,
                                    on_date=a.get("on_date"), at_time=a.get("at_time"))
        if name == "complete_item":
            return scheduler.complete(user_id, a.get("item_id"), a.get("query"), recent_ids,
                                      on_date=a.get("on_date"), at_time=a.get("at_time"), done=a.get("done", True), occurrence_date=a.get("occurrence_date"))
        if name == "list_schedule":
            items = scheduler.schedule(user_id, a["date_from"], a.get("date_to"), _kinds(a),
                                       a.get("include_done", True))
            return {"status": "ok", "count": len(items), "items": items[:60],
                    "truncated": len(items) > 60}
        if name == "search_items":
            return scheduler.search(user_id, a.get("query", ""), a.get("date_from"), a.get("date_to"),
                                    _kinds(a), a.get("limit", 8))
        return {"status": "error", "message": f"Неизвестный инструмент {name}"}
    except (ValueError, KeyError, TypeError) as exc:
        return {"status": "error", "message": f"Некорректные параметры: {exc}. Исправь и вызови снова."}


def touched_ids(name: str, result: dict) -> list[int]:
    """id записей, которые стоит запомнить как «последние упомянутые»."""
    if result.get("status") == "ok":
        if result.get("action") == "deleted":
            return []
        if "item" in result:
            return [result["item"]["id"]]
        if name == "search_items":
            return [r["id"] for r in result.get("results", [])[:3]]
        if name == "list_schedule" and result.get("count", 0) == 1:
            return [result["items"][0]["id"]]
    if result.get("status") in ("conflict", "need_clarification") and "item" in result:
        return [result["item"]["id"]]
    return []

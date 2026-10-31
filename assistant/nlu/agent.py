"""Диалоговый агент: текст пользователя -> LLM с инструментами -> бизнес-логика -> ответ.

Контекст диалога:
- история сообщений пользователя хранится в БД (таблица messages);
- последние упомянутые записи (dialog_state) подставляются в системный промпт,
  поэтому работают фразы вроде «перенеси ее на завтра»;
- краткий журнал вызванных инструментов тоже хранится (role=tool_log) и подставляется в системный
  промпт, чтобы модель помнила, например, параметры встречи, которую пользователь подтверждает после
  конфликта, и варианты, из которых он выбирает. В диалог журнал не попадает: модель копирует такие
  реплики в ответ вместо настоящего вызова инструмента.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Callable

from ..services.scheduler import Scheduler
from ..storage.repository import Repository
from .llm import LLMClient
from .prompts import build_system_prompt
from .tools import TOOL_INDEX, TOOLS, execute, touched_ids

log = logging.getLogger(__name__)
MAX_TOOL_ROUNDS = 6
RECENT_LIMIT = 5
CHANGING_TOOLS = {"create_item", "update_item", "delete_item", "complete_item"}


DROP_KEYS = {"kind_ru", "recurrence", "source"}
SERVICE_LINE = re.compile(r"^.*(?:Служебно|Журнал действий|\b(?:" + "|".join(TOOL_INDEX) + r")\s*\().*$\n?", re.M)
RETRY_HINT = ("Ты написал вызов инструмента текстом, поэтому действие НЕ выполнено. "
              "Вызови инструмент через tools или ответь пользователю обычной фразой.")


def strip_service_text(text: str) -> str:
    """Убрать из ответа модели служебные строки: вызовы инструментов, написанные текстом."""
    if not SERVICE_LINE.search(text):
        return text.strip()
    return re.sub(r"^\s*Ассистент:\s*", "", SERVICE_LINE.sub("", text).strip())


def log_entry(action: dict) -> str:
    """Строка журнала; варианты из ambiguous сохраняются, чтобы следующий шаг мог выбрать по item_id."""
    text = f"{action['tool']}({json.dumps(action['args'], ensure_ascii=False)}) -> {action['status']}"
    candidates = action["result"].get("candidates")
    if candidates:
        text += ": " + "; ".join(f"#{c['id']} {c['title']} {c['date']} {c.get('time') or ''}".strip() for c in candidates)
    return text


def compact(value):
    """Ужать результат инструмента для модели: локальные модели работают с небольшим контекстом."""
    if isinstance(value, dict):
        return {k: compact(v) for k, v in value.items() if k not in DROP_KEYS and v is not None}
    if isinstance(value, list):
        return [compact(v) for v in value]
    return value


@dataclass
class AgentReply:
    reply: str
    actions: list[dict] = field(default_factory=list)

    @property
    def changed(self) -> bool:
        return any(a["tool"] in CHANGING_TOOLS and a["status"] == "ok" for a in self.actions)


class Assistant:
    def __init__(self, repo: Repository, scheduler: Scheduler, llm_factory: Callable[[], LLMClient],
                 timezone: str = "Europe/Moscow", history_messages: int = 12, meeting_minutes: int = 60):
        self.repo = repo
        self.scheduler = scheduler
        self.llm_factory = llm_factory
        self.timezone = timezone
        self.history_messages = history_messages
        self.meeting_minutes = meeting_minutes

    def _history(self, user_id: int) -> tuple[list[dict], list[str]]:
        """Последние реплики диалога и журнал вызовов отдельно."""
        msgs, journal = [], []
        for m in self.repo.recent_messages(user_id, self.history_messages):
            if m["role"] == "tool_log":
                journal.append(m["content"])
                continue
            content = strip_service_text(m["content"])
            if content:
                msgs.append({"role": m["role"], "content": content})
        return msgs, journal

    def handle(self, user_id: int, text: str) -> AgentReply:
        text = (text or "").strip()
        if not text:
            return AgentReply("Я ничего не расслышал. Повторите, пожалуйста.")
        llm = self.llm_factory()  # ConfigError, если нет ключа, - до записи в историю
        user = self.repo.get_user(user_id) or {"name": "пользователь"}
        recent_ids = self.repo.get_recent_ids(user_id)
        recent_items = self.repo.get_items(user_id, recent_ids)
        history, journal = self._history(user_id)
        system = build_system_prompt(user["name"], self.scheduler.now(), self.timezone,
                                     [i.short() for i in recent_items], self.meeting_minutes, journal)
        messages = [{"role": "system", "content": system}, *history, {"role": "user", "content": text}]

        actions: list[dict] = []
        reply = None
        retried = False
        for _ in range(MAX_TOOL_ROUNDS):
            resp = llm.chat(messages, TOOLS)
            if not resp.tool_calls:
                raw = (resp.content or "").strip()
                clean = strip_service_text(raw)
                if clean != raw and not actions:  # вызов текстом: действие не выполнено, словам модели верить нельзя
                    if retried:
                        break
                    retried = True
                    messages.append({"role": "system", "content": RETRY_HINT})
                    continue
                reply = clean
                break
            messages.append(resp.as_message())
            for tc in resp.tool_calls:
                if tc.error:  # модель прислала битый JSON - просим повторить вызов
                    result = {"status": "error",
                              "message": f"Аргументы не разобраны ({tc.error}). Повтори вызов с корректным JSON."}
                else:
                    result = execute(self.scheduler, user_id, tc.name, tc.arguments, recent_ids)
                log.info("tool %s %s -> %s", tc.name, tc.arguments, result.get("status"))
                ids = touched_ids(tc.name, result)
                if ids:
                    recent_ids = list(dict.fromkeys(ids + recent_ids))[:RECENT_LIMIT]
                if result.get("action") == "deleted":
                    recent_ids = [i for i in recent_ids if i != result["item"]["id"]]
                actions.append({"tool": tc.name, "args": tc.arguments, "status": result.get("status"),
                                "result": result})
                messages.append({"role": "tool", "tool_call_id": tc.id, "name": tc.name,
                                 "content": json.dumps(compact(result), ensure_ascii=False, default=str)})
        if reply is None:
            reply = "Не получилось выполнить запрос. Попробуйте сформулировать иначе."
        if not reply:
            reply = "Готово."

        self.repo.set_recent_ids(user_id, recent_ids)
        self.repo.add_message(user_id, "user", text)
        if actions:
            self.repo.add_message(user_id, "tool_log", "; ".join(log_entry(a) for a in actions)[:2000])
        self.repo.add_message(user_id, "assistant", reply)
        return AgentReply(reply, actions)

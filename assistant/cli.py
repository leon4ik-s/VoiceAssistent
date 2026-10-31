"""Командная строка.

    python -m assistant.cli import-csv [--user Имя] [--path файл.csv] [--no-embed]
    python -m assistant.cli embed [--user Имя]
    python -m assistant.cli export-ics [--user Имя] --out calendar.ics
    python -m assistant.cli import-ics файл.ics [--user Имя]
    python -m assistant.cli stats
    python -m assistant.cli check [--full]   проверить настройки (с --full - реальные запросы к сервисам)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import Settings
from .providers import build_context, describe_error
from .services import ics
from .services.importer import import_csv


def full_check(ctx, settings) -> int:
    """По одному маленькому реальному запросу к каждому сервису (стоимость - доли копейки)."""
    import io
    import wave

    from .nlu.tools import TOOLS

    ok = True

    def step(name, fn):
        nonlocal ok
        try:
            print(f"{name}: OK - {fn()}")
        except Exception as exc:
            ok = False
            print(f"{name}: ОШИБКА - {describe_error(exc, settings)[1]}")

    def llm():
        r = ctx.providers.llm().chat([
            {"role": "system", "content": "Сегодня 2026-10-30. Используй инструменты."},
            {"role": "user", "content": "Создай задачу «проверка связи» на завтра"},
        ], TOOLS)
        if r.tool_calls:
            return f"модель вызвала {r.tool_calls[0].name}({r.tool_calls[0].arguments})"
        return f"модель ответила текстом без вызова инструмента: {r.content!r}"

    def emb():
        e = ctx.providers.embedder
        if e is None:
            return "эмбеддинги не настроены"
        v = e.embed(["проверка связи"])[0]
        q = e.embed_query(["проверка"])[0] if hasattr(e, "embed_query") else v
        return f"векторы длины {len(v)} и {len(q)}"

    def stt():
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000); w.writeframes(b"\0\0" * 16000)
        return f"распознано {ctx.providers.stt().transcribe(buf.getvalue(), 'check.wav', 'audio/wav')!r} (тишина)"

    def tts():
        return f"получено {len(ctx.providers.tts().synthesize('Проверка связи'))} байт аудио"

    step("Языковая модель", llm)
    step("Эмбеддинги", emb)
    if ctx.providers.stt_server:
        step("Распознавание", stt)
    if ctx.providers.tts_server:
        step("Озвучка", tts)
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    settings = Settings.from_env()
    p = argparse.ArgumentParser(prog="python -m assistant.cli")
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("import-csv", help="Импорт истории прошлых запросов из CSV")
    a.add_argument("--user", default=settings.default_user)
    a.add_argument("--path", default=str(settings.history_csv))
    a.add_argument("--no-embed", action="store_true", help="Не считать эмбеддинги сейчас")

    e = sub.add_parser("embed", help="Досчитать эмбеддинги для поиска по смыслу")
    e.add_argument("--user", default=settings.default_user)

    x = sub.add_parser("export-ics", help="Выгрузить записи в .ics")
    x.add_argument("--user", default=settings.default_user)
    x.add_argument("--out", default="calendar.ics")

    i = sub.add_parser("import-ics", help="Загрузить события из .ics")
    i.add_argument("path")
    i.add_argument("--user", default=settings.default_user)

    sub.add_parser("stats", help="Пользователи и количество записей")
    c = sub.add_parser("check", help="Проверить настройки провайдера и распознавания речи")
    c.add_argument("--full", action="store_true",
                   help="Сделать по одному реальному запросу к модели, эмбеддингам, распознаванию и озвучке")
    args = p.parse_args(argv)

    ctx = build_context(settings)

    def embed(user_id: int) -> None:
        if ctx.search.index is None:
            print("Модель эмбеддингов не задана: поиск работает только по словам.")
            return
        print(f"Считаю эмбеддинги моделью {settings.active_embed_model} (локально это может занять пару минут)...")
        try:
            n = ctx.search.ensure_embeddings(user_id)
        except Exception as exc:
            print("Не получилось:", describe_error(exc, settings)[1])
            print("Записи загружены; эмбеддинги досчитаются при следующем запуске сервера или командой embed.")
            return
        print(f"Эмбеддинги готовы для {n} записей (запросов к модели: {ctx.search.index.api_calls}, "
              f"остальное из кэша).")

    if args.cmd == "check":
        h = ctx.providers.health()
        print(f"Провайдер: {settings.provider}, модель: {settings.chat_model}, эмбеддинги: {settings.active_embed_model}")
        print("Языковая модель:", "OK" if h.get("ok") else h.get("message"))
        print("Серверное распознавание:", "есть" if ctx.providers.stt_server else "нет (будет использоваться браузер)")
        print("Серверная озвучка:", "есть" if ctx.providers.tts_server else "нет (озвучивает браузер)")
        if not args.full or not h.get("ok"):
            return 0 if h.get("ok") else 1
        return full_check(ctx, settings)

    if args.cmd == "stats":
        for u in ctx.repo.list_users():
            print(f"{u['id']}: {u['name']} - записей {ctx.repo.count_items(u['id'])}, "
                  f"из истории {ctx.repo.count_items(u['id'], 'import')}")
        print(f"Векторов в кэше: {ctx.repo.count_vectors()}")
        return 0

    user = ctx.repo.get_or_create_user(args.user)
    if args.cmd == "import-csv":
        stats = import_csv(ctx.repo, user["id"], Path(args.path), settings.meeting_default_minutes)
        print(f"Пользователь «{user['name']}»: строк {stats['rows']}, добавлено {stats['inserted']}, "
              f"уже были {stats['skipped_existing']}, с ошибками {stats['invalid']}")
        if not args.no_embed:
            embed(user["id"])
    elif args.cmd == "embed":
        embed(user["id"])
    elif args.cmd == "export-ics":
        Path(args.out).write_bytes(ics.export_ics(ctx.repo.list_items(user["id"]), settings.timezone,
                                                  settings.meeting_default_minutes))
        print(f"Сохранено: {args.out}")
    elif args.cmd == "import-ics":
        print(ics.import_ics(ctx.repo, user["id"], Path(args.path).read_bytes(), settings.timezone))
    return 0


if __name__ == "__main__":
    sys.exit(main())

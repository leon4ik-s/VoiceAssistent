"""Веб-слой: FastAPI + одна HTML-страница (голос, текст, история диалога, расписание)."""
from __future__ import annotations

import base64
import logging
import threading
from contextlib import asynccontextmanager
from datetime import date, timedelta
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel

from ..config import ConfigError, Settings
from ..nlu.agent import strip_service_text
from ..providers import AppContext, Providers, build_context, describe_error
from ..services import ics
from ..services.importer import import_csv
from ..yandex_cloud import YandexAPIError

log = logging.getLogger(__name__)
STATIC = Path(__file__).parent / "static"
MAX_AUDIO_BYTES = 25 * 1024 * 1024


class ChatIn(BaseModel):
    user_id: int
    text: str
    speak: bool = False


class UserIn(BaseModel):
    name: str


class TtsIn(BaseModel):
    text: str


class UserRef(BaseModel):
    user_id: int


def create_app(settings: Settings | None = None, providers: Providers | None = None, now_fn=None,
               startup_import: bool = True) -> FastAPI:
    settings = settings or Settings.from_env()
    ctx: AppContext = build_context(settings, providers, now_fn)

    def friendly_error(exc: Exception) -> tuple[int, str]:
        return describe_error(exc, settings)

    def warm_embeddings(user_id: int) -> None:
        if ctx.search.index is None:
            return

        def run():
            try:
                n = ctx.search.ensure_embeddings(user_id)
                log.info("Эмбеддинги готовы для %s записей пользователя %s", n, user_id)
            except Exception as exc:  # не мешаем работе приложения; повторим, когда модель станет доступна
                app.state.embed_retry.add(user_id)
                log.warning("Не удалось посчитать эмбеддинги: %s", friendly_error(exc)[1])

        t = threading.Thread(target=run, daemon=True, name="embeddings")
        app.state.embed_threads.append(t)
        t.start()

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        user = ctx.repo.get_or_create_user(settings.default_user)
        if startup_import and settings.import_csv_on_start and settings.history_csv.exists():
            stats = import_csv(ctx.repo, user["id"], settings.history_csv, settings.meeting_default_minutes)
            log.info("Импорт истории %s: %s", settings.history_csv.name, stats)
            warm_embeddings(user["id"])
        health = ctx.providers.health()
        if not health.get("ok"):
            log.warning("Языковая модель недоступна: %s", health.get("message"))
        yield

    app = FastAPI(title="Голосовой ассистент", lifespan=lifespan)
    app.state.ctx = ctx
    app.state.embed_threads = []
    app.state.embed_retry = set()

    @app.exception_handler(ConfigError)
    async def _config_error(_req: Request, exc: ConfigError):
        return JSONResponse({"error": str(exc)}, status_code=503)

    @app.exception_handler(YandexAPIError)
    async def _yandex_error(_req: Request, exc: YandexAPIError):
        code, msg = friendly_error(exc)
        log.warning("Yandex Cloud: %s", exc)
        return JSONResponse({"error": msg}, status_code=code)

    try:
        import openai

        @app.exception_handler(openai.OpenAIError)
        async def _openai_error(_req: Request, exc: Exception):
            code, msg = friendly_error(exc)
            log.warning("LLM API: %s", exc)
            return JSONResponse({"error": msg}, status_code=code)
    except ImportError:  # pragma: no cover
        pass

    def user_or_404(user_id: int) -> dict:
        u = ctx.repo.get_user(user_id)
        if not u:
            raise HTTPException(404, "Пользователь не найден")
        return u

    def speak(text: str) -> tuple[str | None, str | None]:
        try:
            audio = ctx.providers.tts().synthesize(text)
            return base64.b64encode(audio).decode("ascii"), None
        except Exception as exc:
            return None, friendly_error(exc)[1]

    def chat_payload(user_id: int, text: str, want_audio: bool) -> dict:
        res = ctx.assistant.handle(user_id, text)
        out = {
            "reply": res.reply,
            "changed": res.changed,
            "actions": [{"tool": a["tool"], "status": a["status"]} for a in res.actions],
        }
        if want_audio:
            out["audio"], out["tts_error"] = speak(res.reply)
        return out

    # ------------------------------------------------------------------ страницы
    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(STATIC / "index.html")

    @app.get("/api/status")
    def status():
        health = ctx.providers.health()
        if health.get("ok") and app.state.embed_retry:  # Ollama запустили позже - досчитываем эмбеддинги
            for uid in list(app.state.embed_retry):
                app.state.embed_retry.discard(uid)
                warm_embeddings(uid)
        return {
            "provider": settings.provider,
            "model": settings.chat_model,
            "llm_ok": bool(health.get("ok")),
            "message": health.get("message"),
            "timezone": settings.timezone,
            "now": ctx.scheduler.now().isoformat(timespec="minutes"),
            "users": ctx.repo.list_users(),
            "semantic_search": ctx.search.index is not None,
            "stt_server": ctx.providers.stt_server,
            "tts_server": ctx.providers.tts_server,
            "stt_max_seconds": ctx.providers.stt_max_seconds,
            "language": settings.stt_language,
            "embeddings": ctx.search.index.progress if ctx.search.index else None,
        }

    # ------------------------------------------------------------------ пользователи
    @app.get("/api/users")
    def users():
        return ctx.repo.list_users()

    @app.post("/api/users")
    def add_user(body: UserIn):
        name = body.name.strip()
        if not name or len(name) > 60:
            raise HTTPException(400, "Имя должно быть от 1 до 60 символов")
        return ctx.repo.get_or_create_user(name)

    @app.get("/api/history")
    def history(user_id: int, limit: int = 50):
        user_or_404(user_id)
        msgs = ctx.repo.recent_messages(user_id, limit * 2)
        shown = [{**m, "content": strip_service_text(m["content"])} for m in msgs if m["role"] in ("user", "assistant")]
        return [m for m in shown if m["content"]][-limit:]

    # ------------------------------------------------------------------ диалог
    @app.post("/api/chat")
    def chat(body: ChatIn):
        user_or_404(body.user_id)
        return chat_payload(body.user_id, body.text, body.speak)

    @app.post("/api/voice")
    def voice(user_id: int = Form(...), speak_reply: bool = Form(False), audio: UploadFile = File(...)):
        user_or_404(user_id)
        data = audio.file.read()
        if not data:
            raise HTTPException(400, "Пустая запись")
        if len(data) > MAX_AUDIO_BYTES:
            raise HTTPException(413, "Запись слишком длинная (больше 25 МБ)")
        try:
            text = ctx.providers.stt().transcribe(data, audio.filename or "", audio.content_type)
        except Exception as exc:  # STT не установлен или упал - интерфейс переключится на браузер
            code, msg = friendly_error(exc)
            if code == 500:
                code, msg = 502, f"Не удалось распознать речь на сервере: {exc}"
            log.warning("STT: %s", exc)
            return JSONResponse({"error": msg, "stt_failed": True}, status_code=code)
        if not text:
            return {"transcript": "", "reply": "Не удалось распознать речь, попробуйте еще раз.", "changed": False}
        out = chat_payload(user_id, text, speak_reply)
        out["transcript"] = text
        return out

    @app.post("/api/tts")
    def tts(body: TtsIn):
        audio = ctx.providers.tts().synthesize(body.text)
        return Response(audio, media_type="audio/mpeg")

    # ------------------------------------------------------------------ расписание
    @app.get("/api/schedule")
    def schedule(user_id: int, date_from: str | None = None, date_to: str | None = None, kind: str | None = None):
        user_or_404(user_id)
        today = ctx.scheduler.now().date()
        df = date_from or today.isoformat()
        dt = date_to or (date.fromisoformat(df) + timedelta(days=6)).isoformat()
        try:
            return ctx.scheduler.schedule(user_id, df, dt, [kind] if kind else None)
        except ValueError as exc:
            raise HTTPException(400, str(exc))

    @app.get("/api/upcoming")
    def upcoming(user_id: int, hours: int = 24):
        user_or_404(user_id)
        return ctx.scheduler.upcoming(user_id, hours)

    @app.post("/api/items/{item_id}/complete")
    def complete(item_id: int, user_id: int, done: bool = True, occurrence_date: str | None = None):
        user_or_404(user_id)
        res = ctx.scheduler.complete(user_id, item_id, done=done, occurrence_date=occurrence_date)
        if res["status"] != "ok":
            raise HTTPException(404, res["message"])
        return res

    @app.delete("/api/items/{item_id}")
    def delete(item_id: int, user_id: int):
        user_or_404(user_id)
        res = ctx.scheduler.delete(user_id, item_id)
        if res["status"] != "ok":
            raise HTTPException(404, res["message"])
        return res

    # ------------------------------------------------------------------ импорт / экспорт
    @app.get("/api/export.ics")
    def export(user_id: int):
        user_or_404(user_id)
        data = ics.export_ics(ctx.repo.list_items(user_id), settings.timezone, settings.meeting_default_minutes)
        return Response(data, media_type="text/calendar; charset=utf-8",
                        headers={"Content-Disposition": f'attachment; filename="calendar_user{user_id}.ics"'})

    @app.post("/api/import/ics")
    def import_ics_file(user_id: int = Form(...), file: UploadFile = File(...)):
        user_or_404(user_id)
        try:
            stats = ics.import_ics(ctx.repo, user_id, file.file.read(), settings.timezone)
        except ValueError as exc:
            raise HTTPException(400, f"Не удалось прочитать .ics: {exc}")
        warm_embeddings(user_id)
        return stats

    @app.post("/api/import/history")
    def import_history(body: UserRef):
        user_or_404(body.user_id)
        if not settings.history_csv.exists():
            raise HTTPException(404, f"Файл истории не найден: {settings.history_csv.name}")
        stats = import_csv(ctx.repo, body.user_id, settings.history_csv, settings.meeting_default_minutes)
        warm_embeddings(body.user_id)
        return stats

    return app

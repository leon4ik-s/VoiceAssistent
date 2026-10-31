"""HTTP API: страница, ошибки без ключа, чат, голос, экспорт."""
import pytest
from fastapi.testclient import TestClient

from assistant.web.app import create_app

from conftest import NOW, ScriptedLLM


@pytest.fixture
def client(settings, providers):
    app = create_app(settings, providers, now_fn=lambda: NOW)
    with TestClient(app) as c:
        yield c
    app.state.ctx.db.close()


@pytest.fixture
def no_llm_client(settings, no_key_providers):
    app = create_app(settings, no_key_providers, now_fn=lambda: NOW)
    with TestClient(app) as c:
        yield c
    app.state.ctx.db.close()


def default_user(c):
    return c.get("/api/users").json()[0]["id"]


def test_index_and_status(client):
    r = client.get("/")
    assert r.status_code == 200 and "Голосовой ассистент" in r.text
    st = client.get("/api/status").json()
    assert st["provider"] == "local" and st["llm_ok"] is True and st["message"] is None
    assert st["users"][0]["name"] == "Основной"


def test_without_llm_returns_readable_error(no_llm_client):
    c = no_llm_client
    uid = default_user(c)
    st = c.get("/api/status").json()
    assert st["llm_ok"] is False and "ollama" in st["message"].lower()
    assert st["stt_server"] is False and st["tts_server"] is False   # интерфейс включит браузерные режимы
    r = c.post("/api/chat", json={"user_id": uid, "text": "Привет"})
    assert r.status_code == 503 and "ollama" in r.json()["error"].lower()
    r = c.post("/api/voice", data={"user_id": uid}, files={"audio": ("a.webm", b"x" * 2000, "audio/webm")})
    assert r.status_code == 503 and r.json()["stt_failed"] is True
    assert c.get(f"/api/schedule?user_id={uid}").status_code == 200   # расписание работает без модели


def test_ollama_connection_error_is_explained(settings, providers):
    import openai

    exc = openai.APIConnectionError.__new__(openai.APIConnectionError)
    Exception.__init__(exc, "Connection error.")
    exc.message = "Connection error."

    class Down:
        def chat(self, messages, tools):
            raise exc

    providers.llm = lambda: Down()
    app = create_app(settings, providers, now_fn=lambda: NOW)
    with TestClient(app) as c:
        r = c.post("/api/chat", json={"user_id": default_user(c), "text": "Привет"})
    app.state.ctx.db.close()
    assert r.status_code == 503 and "ollama serve" in r.json()["error"]


def test_stt_crash_suggests_browser(settings, providers):
    class Broken:
        def transcribe(self, *a, **k):
            raise RuntimeError("av: cannot decode")

    providers.stt = lambda: Broken()
    app = create_app(settings, providers, now_fn=lambda: NOW)
    with TestClient(app) as c:
        r = c.post("/api/voice", data={"user_id": default_user(c)},
                   files={"audio": ("a.webm", b"x" * 2000, "audio/webm")})
    app.state.ctx.db.close()
    assert r.status_code == 502 and r.json()["stt_failed"] is True


def test_chat_and_voice(client, llm_holder):
    uid = default_user(client)
    llm_holder["llm"] = ScriptedLLM([
        [("create_item", {"kind": "reminder", "title": "Позвонить Ивану", "date": "2026-10-31", "time": "09:00"})],
        "Напомню завтра в 9 утра позвонить Ивану.",
    ])
    r = client.post("/api/voice", data={"user_id": uid, "speak_reply": "true"},
                    files={"audio": ("speech.webm", b"x" * 2000, "audio/webm")})
    body = r.json()
    assert r.status_code == 200 and body["transcript"] and body["changed"] and body["audio"]
    items = client.get(f"/api/schedule?user_id={uid}&date_from=2026-10-31&date_to=2026-10-31").json()
    assert [i["title"] for i in items] == ["Позвонить Ивану"]

    llm_holder["llm"] = ScriptedLLM(["Завтра в 9 звонок Ивану."])
    r = client.post("/api/chat", json={"user_id": uid, "text": "Что у меня завтра?"})
    assert r.json()["reply"] == "Завтра в 9 звонок Ивану." and "audio" not in r.json()
    hist = client.get(f"/api/history?user_id={uid}").json()
    assert [m["role"] for m in hist] == ["user", "assistant", "user", "assistant"]


def test_items_users_and_ics(client):
    uid = default_user(client)
    ctx = client.app.state.ctx
    item = ctx.scheduler.create(uid, "task", "Отчет", "2026-10-31", "12:00")["item"]
    assert client.post(f"/api/items/{item['id']}/complete?user_id={uid}").json()["item"]["status"] == "done"
    other = client.post("/api/users", json={"name": "Гость"}).json()
    assert client.delete(f"/api/items/{item['id']}?user_id={other['id']}").status_code == 404
    r = client.get(f"/api/export.ics?user_id={uid}")
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/calendar") and b"VEVENT" in r.content
    r = client.post("/api/import/ics", data={"user_id": other["id"]}, files={"file": ("c.ics", r.content, "text/calendar")})
    assert r.json()["created"] == 1
    assert client.delete(f"/api/items/{item['id']}?user_id={uid}").status_code == 200


def test_history_import_on_startup(settings, providers):
    settings.import_csv_on_start = True
    for _ in range(2):  # второй запуск не создает дублей
        app = create_app(settings, providers, now_fn=lambda: NOW)
        with TestClient(app) as c:
            uid = default_user(c)
            assert app.state.ctx.repo.count_items(uid, "import") == 1000
        for t in app.state.embed_threads:
            t.join(timeout=10)
        app.state.ctx.db.close()


def test_history_hides_service_text(client):
    uid = client.post("/api/users", json={"name": "Тест"}).json()["id"]
    repo = client.app.state.ctx.repo
    repo.add_message(uid, "user", "Удали встречу")
    repo.add_message(uid, "tool_log", 'delete_item({"item_id": 1}) -> ok')
    repo.add_message(uid, "assistant", 'Служебно, ранее вызванные инструменты: delete_item({"item_id": 1}) -> ok\n\n Встреча удалена.')
    repo.add_message(uid, "assistant", 'delete_item({"item_id": 1})')
    assert [m["content"] for m in client.get(f"/api/history?user_id={uid}").json()] == ["Удали встречу", "Встреча удалена."]

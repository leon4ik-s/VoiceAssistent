"""Провайдер Yandex Cloud: формат запросов к API (через настоящий клиент openai и urllib), разбор ответов, ошибки.

Вместо Yandex Cloud поднимается локальный HTTP-сервер, который записывает запросы и отвечает
в формате документации AI Studio и SpeechKit.
"""
import io
import json
import threading
import urllib.parse
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import pytest
from fastapi.testclient import TestClient

from assistant.config import ConfigError, Settings
from assistant.nlu.tools import TOOLS
from assistant.providers import build_providers, describe_error
from assistant.voice.yandex_speech import prepare_audio, wav_to_lpcm
from assistant.web.app import create_app
from assistant.yandex_cloud import YandexAPIError, YandexEmbedder, yandex_error_message

from conftest import NOW

FOLDER = "b1gtestfolder"
KEY = "AQVNtestkey"


class FakeYandex:
    """Записывает запросы; ответы задаются по пути. По умолчанию - успешные ответы в формате API."""

    def __init__(self):
        self.requests = []
        self.fail: dict[str, tuple[int, str]] = {}
        self.chat_queue: list[dict] = []  # ответы на /v1/chat/completions по очереди
        outer = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_POST(self):
                body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
                path = urllib.parse.urlparse(self.path)
                outer.requests.append({"path": path.path, "query": dict(urllib.parse.parse_qsl(path.query)),
                                       "headers": {k.lower(): v for k, v in self.headers.items()}, "body": body})
                for prefix, (code, text) in outer.fail.items():
                    if path.path.startswith(prefix):
                        return self._send(code, text.encode(), "application/json")
                if path.path == "/v1/chat/completions":
                    return self._json(outer.chat_queue.pop(0) if outer.chat_queue else _chat_tool_call())
                if path.path == "/v1/embeddings":
                    req = json.loads(body)
                    seed = sum(map(ord, req["input"])) % 7 + 1
                    return self._json({"object": "list", "model": req["model"],
                                       "data": [{"object": "embedding", "index": 0, "embedding": [seed, 1.0, 0.5]}]})
                if path.path == "/speech/v1/stt:recognize":
                    return self._json({"result": "встреча с иваном завтра в три"})
                if path.path == "/speech/v1/tts:synthesize":
                    return self._send(200, b"ID3-mp3-bytes", "audio/mpeg")
                self._send(404, b"{}", "application/json")

            def _json(self, obj):
                self._send(200, json.dumps(obj, ensure_ascii=False).encode(), "application/json")

            def _send(self, code, data, ctype):
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.server.server_port}"

    def last(self, path):
        return [r for r in self.requests if r["path"] == path][-1]


def _chat_tool_call():
    return {
        "id": "chatcmpl-1", "object": "chat.completion", "created": 0, "model": f"gpt://{FOLDER}/yandexgpt-5.1",
        "choices": [{"index": 0, "finish_reason": "tool_calls", "message": {
            "role": "assistant", "content": None,
            "tool_calls": [{"id": "call_1", "type": "function", "function": {
                "name": "create_item",
                "arguments": json.dumps({"kind": "meeting", "title": "Встреча с Иваном", "date": "2026-10-31",
                                         "time": "15:00"}, ensure_ascii=False)}}]}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


def _chat_text(text):
    return {"id": "chatcmpl-2", "object": "chat.completion", "created": 0, "model": "m",
            "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": text}}]}


@pytest.fixture
def yc():
    srv = FakeYandex()
    yield srv
    srv.server.shutdown()


def ysettings(yc, tmp_path, **kw) -> Settings:
    return Settings(provider="yandex", yandex_api_key=KEY, yandex_folder_id=FOLDER,
                    yandex_llm_base_url=f"{yc.url}/v1", yandex_stt_url=f"{yc.url}/speech/v1/stt:recognize",
                    yandex_tts_url=f"{yc.url}/speech/v1/tts:synthesize", yandex_embed_rps=1000,
                    db_path=tmp_path / "y.db", import_csv_on_start=False, **kw)


def wav_bytes(seconds=1.0, rate=16000, channels=1) -> bytes:
    t = np.arange(int(seconds * rate)) / rate
    x = (0.3 * np.sin(2 * np.pi * 300 * t) * 32767).astype("<i2")
    if channels == 2:
        x = np.repeat(x, 2)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(channels); w.setsampwidth(2); w.setframerate(rate); w.writeframes(x.tobytes())
    return buf.getvalue()


# ---------------------------------------------------------------- настройки
def test_defaults_and_uris():
    s = Settings(yandex_folder_id=FOLDER)
    assert s.provider == "yandex" and s.resolved_stt == "yandex" and s.resolved_tts == "yandex"
    assert s.yandex_gpt_uri == f"gpt://{FOLDER}/yandexgpt-5.1"
    assert s.yandex_emb_uri("text-search-doc/latest") == f"emb://{FOLDER}/text-search-doc/latest"
    s.yandex_gpt_model = "gpt://other/aliceai-llm-flash"
    assert s.yandex_gpt_uri == "gpt://other/aliceai-llm-flash"


def test_without_keys_gives_clear_message(tmp_path):
    p = build_providers(Settings(db_path=tmp_path / "x.db"))
    for fn in (p.llm, p.stt, p.tts):
        with pytest.raises(ConfigError, match="YANDEX_API_KEY"):
            fn()
    assert p.health()["ok"] is False and "YANDEX_FOLDER_ID" in p.health()["message"]
    assert p.stt_server is False and p.tts_server is False and p.embedder is None


# ---------------------------------------------------------------- YandexGPT (OpenAI-совместимый API)
def test_chat_request_format_and_tool_call(yc, tmp_path):
    p = build_providers(ysettings(yc, tmp_path))
    resp = p.llm().chat([{"role": "user", "content": "Встреча с Иваном завтра в 15"}], TOOLS)
    req = yc.last("/v1/chat/completions")
    body = json.loads(req["body"])
    assert req["headers"]["authorization"] == f"Bearer {KEY}"
    assert req["headers"]["openai-project"] == FOLDER
    assert body["model"] == f"gpt://{FOLDER}/yandexgpt-5.1"
    assert body["tool_choice"] == "auto" and {t["function"]["name"] for t in body["tools"]} >= {"create_item"}
    assert body["temperature"] == pytest.approx(0.1)
    assert resp.tool_calls[0].name == "create_item" and resp.tool_calls[0].arguments["time"] == "15:00"


def test_full_dialog_through_yandex_api(yc, tmp_path):
    """Веб-API -> агент -> YandexGPT (tools) -> бизнес-логика -> SpeechKit TTS."""
    app = create_app(ysettings(yc, tmp_path), now_fn=lambda: NOW)
    with TestClient(app) as c:
        uid = c.get("/api/users").json()[0]["id"]
        st = c.get("/api/status").json()
        assert st["provider"] == "yandex" and st["llm_ok"] and st["tts_server"] and st["stt_max_seconds"] == 29
        yc.chat_queue = [_chat_tool_call(), _chat_text("Записал встречу с Иваном на завтра в 15:00.")]
        r = c.post("/api/chat", json={"user_id": uid, "text": "Встреча с Иваном завтра в 15", "speak": True})
        body = r.json()
        assert r.status_code == 200 and body["changed"] and body["reply"].startswith("Записал")
        assert body["audio"]                                   # озвучка пришла от SpeechKit
        items = c.get(f"/api/schedule?user_id={uid}&date_from=2026-10-31&date_to=2026-10-31").json()
        assert [(i["title"], i["time"]) for i in items] == [("Встреча с Иваном", "15:00")]
        # второй запрос к модели содержит вызов инструмента и его результат (role=tool, tool_call_id)
        second = json.loads([q for q in yc.requests if q["path"] == "/v1/chat/completions"][-1]["body"])
        assert [m["role"] for m in second["messages"]][-2:] == ["assistant", "tool"]
        assert second["messages"][-2]["tool_calls"][0]["id"] == second["messages"][-1]["tool_call_id"] == "call_1"
        assert json.loads(second["messages"][-1]["content"])["status"] == "ok"
        assert yc.last("/speech/v1/tts:synthesize")["headers"]["authorization"] == f"Api-Key {KEY}"
    app.state.ctx.db.close()


# ---------------------------------------------------------------- эмбеддинги
def test_embeddings_doc_and_query_models(yc, tmp_path):
    settings = ysettings(yc, tmp_path)
    from assistant.providers import build_context

    ctx = build_context(settings, now_fn=lambda: NOW)
    uid = ctx.repo.get_or_create_user("Анна")["id"]
    for t in ["Отправить договор Ивану", "Купить хлеб", "Согласовать договор с Иваном"]:
        ctx.scheduler.create(uid, "task", t, "2026-11-01")
    ctx.search.ensure_embeddings(uid)
    emb = [json.loads(r["body"]) for r in yc.requests if r["path"] == "/v1/embeddings"]
    assert len(emb) == 3                                   # API принимает одну строку за запрос
    assert all(isinstance(e["input"], str) and e["encoding_format"] == "float" for e in emb)
    assert {e["model"] for e in emb} == {f"emb://{FOLDER}/text-search-doc/latest"}
    ctx.search.search(uid, "договор с Иваном")
    q = json.loads(yc.last("/v1/embeddings")["body"])
    assert q["model"] == f"emb://{FOLDER}/text-search-query/latest" and q["input"] == "договор с Иваном"
    # повторный пересчет берет векторы из кэша
    n = len(yc.requests)
    ctx.search.ensure_embeddings(uid)
    assert len(yc.requests) == n
    ctx.db.close()


def test_embedder_throttles_and_retries_429():
    sleeps = []

    class Err(Exception):
        status_code = 429

    class Client:
        def __init__(self):
            self.n = 0
            self.embeddings = self

        def create(self, model, input, encoding_format):
            self.n += 1
            if self.n == 1:
                raise Err("429 Too Many Requests")
            return type("R", (), {"data": [type("D", (), {"embedding": [1.0, 2.0]})()]})()

    e = YandexEmbedder(Client(), FOLDER, "text-search-doc/latest", "text-search-query/latest", rps=10,
                       sleep=sleeps.append)
    assert e.embed(["a", "b"]) == [[1.0, 2.0], [1.0, 2.0]]
    assert 1.0 in sleeps                       # пауза после 429
    assert all(s <= 1.0 for s in sleeps)       # между запросами - не больше 1/rps (плюс пауза 429)
    assert e.model == "yandex:text-search-doc/latest" and e.query_model == "yandex:text-search-query/latest"


# ---------------------------------------------------------------- SpeechKit
def test_stt_request_format_wav(yc, tmp_path):
    p = build_providers(ysettings(yc, tmp_path))
    text = p.stt().transcribe(wav_bytes(1.0), "speech.wav", "audio/wav")
    req = yc.last("/speech/v1/stt:recognize")
    assert text == "встреча с иваном завтра в три"
    assert req["headers"]["authorization"] == f"Api-Key {KEY}"
    assert req["query"] == {"lang": "ru-RU", "topic": "general", "format": "lpcm", "sampleRateHertz": "16000"}
    assert len(req["body"]) == 32000                       # 1 с * 16000 Гц * 2 байта, без WAV-заголовка
    assert "folderId" not in req["query"]                  # для сервисного аккаунта не нужен


def test_stt_ogg_passthrough_and_iam_folder(yc, tmp_path):
    s = ysettings(yc, tmp_path, yandex_iam_token="t1.iam")
    s.yandex_api_key = ""
    p = build_providers(s)
    assert p.stt_server
    p.stt().transcribe(b"OggS" + b"\0" * 100, "a.ogg", "audio/ogg")
    req = yc.last("/speech/v1/stt:recognize")
    assert req["query"]["format"] == "oggopus" and req["query"]["folderId"] == FOLDER  # IAM-токен - нужен folderId
    assert req["headers"]["authorization"] == "Bearer t1.iam"


def test_audio_conversion_and_limits():
    pcm, rate = wav_to_lpcm(wav_bytes(1.0, rate=44100, channels=2))
    assert rate == 16000 and abs(len(pcm) - 32000) <= 2
    pcm, rate = wav_to_lpcm(wav_bytes(0.5, rate=48000))
    assert rate == 48000 and len(pcm) == 48000
    with pytest.raises(ConfigError, match="30 секунд"):
        prepare_audio(wav_bytes(31.0))


def test_stt_unknown_format_without_ffmpeg(monkeypatch):
    import assistant.voice.yandex_speech as ys

    monkeypatch.setattr(ys.shutil, "which", lambda name: None)
    with pytest.raises(ConfigError, match="ffmpeg"):
        prepare_audio(b"\x1aE\xdf\xa3webm-data")


def test_tts_request_format(yc, tmp_path):
    p = build_providers(ysettings(yc, tmp_path, yandex_tts_voice="jane"))
    audio = p.tts().synthesize("Записал встречу на завтра.")
    req = yc.last("/speech/v1/tts:synthesize")
    form = dict(urllib.parse.parse_qsl(req["body"].decode()))
    assert audio == b"ID3-mp3-bytes"
    assert form == {"text": "Записал встречу на завтра.", "lang": "ru-RU", "voice": "jane", "speed": "1.0", "format": "mp3"}
    assert req["headers"]["content-type"] == "application/x-www-form-urlencoded"


# ---------------------------------------------------------------- ошибки
@pytest.mark.parametrize("status,text,expect", [
    (401, '{"error_code":"UNAUTHORIZED","error_message":"Unknown api key"}', "YANDEX_API_KEY"),
    (403, '{"error_code":"PERMISSION_DENIED","error_message":"Permission denied"}', "ai.languageModels.user"),
    (403, '{"error_message":"Billing account is not active"}', "баланс"),
    (402, "{}", "баланс"),
    (429, '{"error_message":"Too many requests"}', "лимит"),
    (400, '{"error":{"message":"Model gpt://x/yandexgpt-lite not found"}}', "YANDEX_GPT_MODEL"),
    (None, "timed out", "Нет связи"),
])
def test_error_messages(status, text, expect):
    assert expect in yandex_error_message(status, text)[1]


def test_speechkit_errors_reach_ui(yc, tmp_path):
    settings = ysettings(yc, tmp_path)
    yc.fail["/speech/v1/stt:recognize"] = (401, '{"error_code":"UNAUTHORIZED","error_message":"Unknown api key"}')
    yc.fail["/speech/v1/tts:synthesize"] = (429, '{"error_message":"Too many requests"}')
    app = create_app(settings, now_fn=lambda: NOW)
    with TestClient(app) as c:
        uid = c.get("/api/users").json()[0]["id"]
        r = c.post("/api/voice", data={"user_id": uid}, files={"audio": ("s.wav", wav_bytes(1.0), "audio/wav")})
        assert r.status_code == 401 and "YANDEX_API_KEY" in r.json()["error"] and r.json()["stt_failed"]
        r = c.post("/api/tts", json={"text": "привет"})
        assert r.status_code == 429 and "лимит" in r.json()["error"]
    app.state.ctx.db.close()


def test_yandexgpt_http_errors_are_explained(yc, tmp_path):
    settings = ysettings(yc, tmp_path)
    yc.fail["/v1/chat/completions"] = (403, '{"error":{"message":"Permission denied to folder"}}')
    p = build_providers(settings)
    with pytest.raises(Exception) as ei:
        p.llm().chat([{"role": "user", "content": "x"}], TOOLS)
    code, msg = describe_error(ei.value, settings)
    assert code == 403 and "ai.languageModels.user" in msg
    yc.fail["/v1/chat/completions"] = (402, '{"error":{"message":"Payment required: balance"}}')
    with pytest.raises(Exception) as ei:
        p.llm().chat([{"role": "user", "content": "x"}], TOOLS)
    assert "баланс" in describe_error(ei.value, settings)[1]


def test_yandex_api_error_type():
    e = YandexAPIError(429, "x", "SpeechKit")
    assert e.status_code == 429 and "SpeechKit" in str(e)

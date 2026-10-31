"""Запуск сервера: python run.py (адрес и порт берутся из .env, по умолчанию http://127.0.0.1:8000)."""
import uvicorn

from assistant.config import Settings

if __name__ == "__main__":
    s = Settings.from_env()
    print(f"Откройте в браузере: http://{s.host}:{s.port}")
    uvicorn.run("assistant.main:app", host=s.host, port=s.port)

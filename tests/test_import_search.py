"""Импорт истории из CSV (идемпотентность, кэш эмбеддингов) и поиск по смыслу."""
import csv
import math

from assistant.services.importer import classify, import_csv
from assistant.services.search import SemanticSearch
from assistant.storage.embeddings import EmbeddingIndex

from conftest import CSV_PATH


def unique_titles() -> int:
    with open(CSV_PATH, encoding="utf-8-sig") as f:
        return len({" ".join(r["text"].split()) for r in csv.DictReader(f)})


def test_csv_import_is_idempotent(ctx, user):
    first = import_csv(ctx.repo, user, CSV_PATH)
    assert first == {"rows": 1000, "inserted": 1000, "skipped_existing": 0, "invalid": 0}
    second = import_csv(ctx.repo, user, CSV_PATH)
    assert second["inserted"] == 0 and second["skipped_existing"] == 1000
    assert ctx.repo.count_items(user) == 1000
    item = ctx.repo.find_by_external(user, "import", "csv:1")
    assert (item.date, item.time, item.title) == ("2024-01-07", "14:45", "Подготовить техническое задание")


def test_classify_kinds():
    assert classify("Демо проекта по хакатону") == "meeting"
    assert classify("Созвон с Альфой") == "meeting"
    assert classify("Провести планёрку по CRM") == "meeting"
    assert classify("Напомнить Анной") == "reminder"
    assert classify("Подготовить отчёт") == "task"


def test_import_per_user(ctx, user, other_user):
    import_csv(ctx.repo, user, CSV_PATH)
    assert ctx.repo.count_items(other_user) == 0
    assert import_csv(ctx.repo, other_user, CSV_PATH)["inserted"] == 1000


def test_embeddings_are_batched_and_cached(ctx, user, embedder):
    import_csv(ctx.repo, user, CSV_PATH)
    index = EmbeddingIndex(ctx.repo, embedder, batch_size=100)
    search = SemanticSearch(ctx.repo, index)
    assert search.ensure_embeddings(user) == 1000
    n = unique_titles()
    assert index.api_calls == math.ceil(n / 100)
    assert sum(len(c) for c in embedder.calls) == n          # каждый уникальный текст ровно один раз
    assert max(len(c) for c in embedder.calls) <= 100
    # повторный импорт и пересчет не обращаются к API
    import_csv(ctx.repo, user, CSV_PATH)
    search.ensure_embeddings(user)
    assert index.api_calls == math.ceil(n / 100)
    assert ctx.repo.count_vectors() == n


def test_semantic_search_over_history(ctx, user):
    import_csv(ctx.repo, user, CSV_PATH)
    res = ctx.scheduler.search(user, "договор с Иваном", limit=5)
    assert res["count"] > 0
    top = res["results"][0]["title"]
    assert "договор" in top.lower() and "иван" in top.lower()

    res = ctx.scheduler.search(user, "демо по хакатону", limit=10)
    titles = [r["title"] for r in res["results"]]
    exact = titles.count("Демо проекта по хакатону")
    assert exact >= 1 and titles[:exact] == ["Демо проекта по хакатону"] * exact
    assert all(r["date"] and r["time"] for r in res["results"])

    # фильтр по датам
    res = ctx.scheduler.search(user, "демо по хакатону", date_from="2024-01-01", date_to="2024-01-31")
    assert res["results"] and all(r["date"].startswith("2024-01") for r in res["results"])


def test_search_without_embeddings_falls_back_to_words(ctx, user):
    ctx.scheduler.create(user, "task", "Отправить договор Иваном", "2026-11-01")
    ctx.scheduler.create(user, "task", "Купить хлеб", "2026-11-01")
    plain = SemanticSearch(ctx.repo, None)
    found = plain.search(user, "договор Иван")
    assert [i.title for i, _ in found] == ["Отправить договор Иваном"]


def test_semantic_component_ranks_by_meaning(ctx, user, embedder):
    """Фейковые векторы: запрос, похожий по смыслу, получает больший балл."""
    for t in ["Созвон с Альфой по бюджету", "Купить молоко", "Обсудить бюджет по договору"]:
        ctx.scheduler.create(user, "task", t, "2026-11-01")
    found = ctx.search.search(user, "бюджет")
    assert {i.title for i, _ in found} == {"Созвон с Альфой по бюджету", "Обсудить бюджет по договору"}

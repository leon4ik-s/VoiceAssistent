"""Повторяющиеся события: проверка правила и разворачивание экземпляров."""
from __future__ import annotations

from datetime import date, datetime, time, timedelta

from dateutil import rrule

from ..storage.models import Item

FREQS = {
    "daily": rrule.DAILY,
    "weekly": rrule.WEEKLY,
    "monthly": rrule.MONTHLY,
    "yearly": rrule.YEARLY,
    "weekdays": rrule.WEEKLY,  # по будням = еженедельно пн-пт
}
WEEKDAY_CODES = ["MO", "TU", "WE", "TH", "FR", "SA", "SU"]
FREQ_RU = {
    "daily": "ежедневно",
    "weekly": "еженедельно",
    "monthly": "ежемесячно",
    "yearly": "ежегодно",
    "weekdays": "по будням",
}
WEEKDAY_RU = ["пн", "вт", "ср", "чт", "пт", "сб", "вс"]


def normalize(rule: dict | None) -> dict | None:
    """Проверить и привести правило к виду {freq, interval, weekdays?, until?, count?}."""
    if not rule:
        return None
    freq = str(rule.get("freq", "")).lower().strip()
    if freq in ("", "none"):
        return None
    if freq not in FREQS:
        raise ValueError(f"Неизвестная периодичность: {freq}")
    out: dict = {"freq": freq, "interval": max(1, int(rule.get("interval") or 1))}
    days = rule.get("weekdays") or []
    if freq == "weekdays":
        days = WEEKDAY_CODES[:5]
    if days:
        days = [str(d).upper()[:2] for d in days]
        bad = [d for d in days if d not in WEEKDAY_CODES]
        if bad:
            raise ValueError(f"Неизвестные дни недели: {bad}")
        out["weekdays"] = days
    if rule.get("until"):
        out["until"] = date.fromisoformat(str(rule["until"])).isoformat()
    if rule.get("count"):
        out["count"] = max(1, int(rule["count"]))
    return out


def to_rrule(item: Item) -> rrule.rrule:
    r = item.recurrence or {}
    kwargs: dict = {"dtstart": item.start_dt(), "interval": r.get("interval", 1)}
    if r.get("weekdays"):
        kwargs["byweekday"] = [getattr(rrule, d) for d in r["weekdays"]]
    if r.get("until"):
        kwargs["until"] = datetime.combine(date.fromisoformat(r["until"]), time(23, 59, 59))
    if r.get("count"):
        kwargs["count"] = r["count"]
    return rrule.rrule(FREQS[r["freq"]], **kwargs)


def occurrence_dates(item: Item, start: date, end: date) -> list[date]:
    """Даты экземпляров записи в интервале [start, end] включительно."""
    if not item.recurrence:
        return [item.start_date] if start <= item.start_date <= end else []
    rule = to_rrule(item)
    lo = datetime.combine(start, time(0, 0))
    hi = datetime.combine(end, time(23, 59, 59))
    return [dt.date() for dt in rule.between(lo, hi, inc=True)]


def last_date(item: Item) -> date | None:
    """Последняя дата серии или None, если серия бесконечна."""
    if not item.recurrence:
        return item.start_date
    r = item.recurrence
    if r.get("count"):
        return list(to_rrule(item))[-1].date()
    if r.get("until"):
        return date.fromisoformat(r["until"])
    return None


def describe(rule: dict | None) -> str:
    if not rule:
        return ""
    parts = [FREQ_RU[rule["freq"]]]
    if rule.get("interval", 1) > 1:
        parts.append(f"каждые {rule['interval']}")
    if rule.get("weekdays") and rule["freq"] != "weekdays":
        parts.append("(" + ", ".join(WEEKDAY_RU[WEEKDAY_CODES.index(d)] for d in rule["weekdays"]) + ")")
    if rule.get("until"):
        parts.append(f"до {rule['until']}")
    if rule.get("count"):
        parts.append(f"{rule['count']} раз")
    return " ".join(parts)


def nearest_occurrence(item: Item, around: date) -> date | None:
    """Ближайший к дате экземпляр (сначала ищем вперед, потом назад)."""
    fwd = occurrence_dates(item, around, around + timedelta(days=400))
    if fwd:
        return fwd[0]
    back = occurrence_dates(item, around - timedelta(days=400), around)
    return back[-1] if back else None

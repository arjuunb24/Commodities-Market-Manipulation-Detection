"""
src/live/calendar.py
====================
Recurring scheduled events that move commodity futures for ordinary reasons.

A sharp move or volume burst at 10:30 ET on a Wednesday in crude oil is the
weekly EIA inventory report, not a pump; a volume spike at 14:30 ET is the
settlement window. The checker treats a flagged window that overlaps one of
these as "likely benign" unless the pattern clearly stands apart from it.

Times are the usual exchange schedule (CME Globex / COMEX / NYMEX / CBOT, LBMA
benchmarks). Event-specific dates (FOMC, monthly USDA WASDE, holidays, contract
rolls) are not tracked; daily windows when such reports usually land are marked
"possible".
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import time
from zoneinfo import ZoneInfo

import pandas as pd

from src.live.datasets import SECTOR

ET = ZoneInfo("America/New_York")
LONDON = ZoneInfo("Europe/London")
WEEKDAYS = (0, 1, 2, 3, 4)
BEFORE_MIN, AFTER_MIN = 3, 15        # a window "overlaps" an event within [t-3 min, t+15 min]


@dataclass(frozen=True)
class Event:
    name: str
    at: time
    tz: ZoneInfo
    scope: tuple[str, ...]           # commodities or sectors ("all" for everything)
    days: tuple[int, ...] = WEEKDAYS
    kind: str = "scheduled release"
    certainty: str = "fixed"         # "fixed" (every listed day) or "possible" (some days only)


EVENTS: list[Event] = [
    Event("US economic data releases (8:30 ET)", time(8, 30), ET, ("all",), certainty="possible"),
    Event("US economic data releases (10:00 ET)", time(10, 0), ET, ("all",), certainty="possible"),
    Event("FOMC statement window (14:00 ET)", time(14, 0), ET, ("all",), certainty="possible"),
    Event("Daily Globex halt starts (17:00 ET)", time(17, 0), ET, ("all",), kind="session"),
    Event("Globex reopen, thin liquidity (18:00 ET)", time(18, 0), ET, ("all",), days=(6, 0, 1, 2, 3), kind="session"),
    # Energy
    Event("NYMEX energy day session open", time(9, 0), ET, ("energy",), kind="session"),
    Event("EIA weekly petroleum status report", time(10, 30), ET, ("crude_oil_wti", "brent_crude"), days=(2,)),
    Event("EIA weekly natural gas storage report", time(10, 30), ET, ("natural_gas",), days=(3,)),
    Event("API weekly crude inventory", time(16, 30), ET, ("crude_oil_wti", "brent_crude"), days=(1,)),
    Event("NYMEX energy settlement window", time(14, 28), ET, ("energy",), kind="settlement"),
    # Metals
    Event("COMEX metals day session open", time(8, 20), ET, ("metals",), kind="session"),
    Event("LBMA gold price (AM auction)", time(10, 30), LONDON, ("gold",), kind="benchmark fix"),
    Event("LBMA gold price (PM auction)", time(15, 0), LONDON, ("gold",), kind="benchmark fix"),
    Event("LBMA silver price auction", time(12, 0), LONDON, ("silver",), kind="benchmark fix"),
    Event("COMEX copper settlement window", time(12, 58), ET, ("copper",), kind="settlement"),
    Event("COMEX silver settlement window", time(13, 24), ET, ("silver",), kind="settlement"),
    Event("COMEX gold settlement window", time(13, 29), ET, ("gold",), kind="settlement"),
    # Grains
    Event("CBOT grains day session open", time(9, 30), ET, ("grains",), kind="session"),
    Event("USDA reports (12:00 ET, e.g. WASDE)", time(12, 0), ET, ("grains",), certainty="possible"),
    Event("CBOT grains close and settlement", time(14, 14), ET, ("grains",), kind="settlement"),
    Event("USDA crop progress (Mondays 16:00 ET)", time(16, 0), ET, ("grains",), days=(0,)),
]


def _applies(ev: Event, commodity: str) -> bool:
    return "all" in ev.scope or commodity in ev.scope or SECTOR.get(commodity) in ev.scope


def events_overlapping(commodity: str, start_utc: pd.Timestamp, end_utc: pd.Timestamp) -> list[dict]:
    """Scheduled events for this commodity whose window overlaps [start, end] (UTC)."""
    start_utc, end_utc = pd.Timestamp(start_utc).tz_convert("UTC"), pd.Timestamp(end_utc).tz_convert("UTC")
    hits = []
    for ev in EVENTS:
        if not _applies(ev, commodity):
            continue
        local_days = pd.date_range((start_utc - pd.Timedelta(days=1)).tz_convert(ev.tz).normalize(),
                                   (end_utc + pd.Timedelta(days=1)).tz_convert(ev.tz).normalize(), freq="D")
        for day in local_days:
            if day.weekday() not in ev.days:
                continue
            at = pd.Timestamp.combine(day.date(), ev.at).tz_localize(ev.tz).tz_convert("UTC")
            lo, hi = at - pd.Timedelta(minutes=BEFORE_MIN), at + pd.Timedelta(minutes=AFTER_MIN)
            if lo <= end_utc and hi >= start_utc:
                hits.append({"event": ev.name, "kind": ev.kind, "certainty": ev.certainty,
                             "at_utc": at.strftime("%Y-%m-%d %H:%M"),
                             "at_local": at.tz_convert(ev.tz).strftime("%a %H:%M %Z")})
    return hits

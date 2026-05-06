"""
Asian session range detection.

Computes, per Asian session window and configurable timezone, the session high and low
for use in liquidity and breakout-style strategies.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from typing import Literal
from zoneinfo import ZoneInfo

import pandas as pd

SessionBoundaryMode = Literal["start_date", "end_date"]


@dataclass(frozen=True, slots=True)
class AsianSessionWindow:
    """
    Local session bounds in ``timezone`` (wall-clock).

    If ``asian_start < asian_end`` (e.g. 00:00–08:00), the window is same calendar day.

    If ``asian_start >= asian_end`` (e.g. 22:00–08:00), the window spans overnight;
    rows are grouped into a **session date** using ``overnight_label``.
    """

    timezone: str
    asian_start: time
    asian_end: time
    overnight_label: SessionBoundaryMode = "start_date"


def _local_time_in_simple_window(t: time, start: time, end: time) -> bool:
    """Same-day window: ``start <= t < end`` (requires ``start < end``)."""
    return start <= t < end


def _asian_session_label_date(
    local: datetime,
    asian_start: time,
    asian_end: time,
    overnight_label: SessionBoundaryMode,
) -> date | None:
    """
    Map a localized timestamp to the **session date** key if it falls inside the window.

    - Same-day window: session date is the local calendar date of the bar.
    - Overnight window: evening leg uses ``local.date()``; morning leg before ``asian_end``
      attaches to the previous calendar day when ``overnight_label=='start_date'``,
      else uses ``local.date()`` for the morning (end_date labeling).
    """
    if asian_start < asian_end:
        if _local_time_in_simple_window(local.time(), asian_start, asian_end):
            return local.date()
        return None

    # Overnight: e.g. 22:00–08:00 local
    t = local.time()
    d = local.date()
    if overnight_label == "end_date":
        # Session keyed by the calendar date on which the window **closes** (e.g. Tuesday 08:00).
        if t >= asian_start:
            return d + timedelta(days=1)
        if t < asian_end:
            return d
        return None
    # start_date: evening leg keeps same calendar day; morning leg rolls back to prior day
    if t >= asian_start:
        return d
    if t < asian_end:
        return d - timedelta(days=1)
    return None


def compute_asian_session_extremes(
    ohlcv: pd.DataFrame,
    window: AsianSessionWindow,
) -> pd.DataFrame:
    """
    Build a dataframe of Asian session high / low per session date.

    Parameters
    ----------
    ohlcv
        DatetimeIndex (tz-aware, UTC recommended) and columns at least ``high``, ``low``.
    window
        Session definition in local timezone.

    Returns
    -------
    DataFrame indexed by ``session_date`` with columns ``asian_high``, ``asian_low``.
    """
    required = {"high", "low"}
    missing = required - set(ohlcv.columns)
    if missing:
        raise ValueError(f"ohlcv missing columns: {sorted(missing)}")

    if not isinstance(ohlcv.index, pd.DatetimeIndex):
        raise TypeError("ohlcv must use a DatetimeIndex")

    if ohlcv.index.tz is None:
        raise ValueError("DatetimeIndex must be timezone-aware (use UTC).")

    tz = ZoneInfo(window.timezone)
    local_index = ohlcv.index.tz_convert(tz)

    labels: list[date | None] = []
    for ts in local_index:
        labels.append(
            _asian_session_label_date(
                ts.to_pydatetime(),
                window.asian_start,
                window.asian_end,
                window.overnight_label,
            )
        )

    labeled = ohlcv.assign(_session_date=labels)
    in_session = labeled["_session_date"].notna()
    trimmed = labeled.loc[in_session].copy()
    trimmed["_session_date"] = trimmed["_session_date"].astype("object")

    grouped = trimmed.groupby("_session_date", sort=True)
    out = grouped.agg(asian_high=("high", "max"), asian_low=("low", "min"))
    out.index.name = "session_date"
    out = out.reset_index()
    return out


def print_sample_asian_days(
    extremes: pd.DataFrame,
    n: int = 5,
) -> None:
    """Print sample rows: date, Asian high, Asian low."""
    head = extremes.head(n)
    print("date       | asian_high | asian_low")
    print("-" * 42)
    for _, row in head.iterrows():
        sd = row["session_date"]
        print(f"{sd} | {row['asian_high']:10.5f} | {row['asian_low']:9.5f}")

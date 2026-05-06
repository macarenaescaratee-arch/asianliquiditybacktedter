"""Shared types for the live EURUSD bot."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal

import pandas as pd

Direction = Literal["long", "short"]


@dataclass
class Candle:
    """Single OHLCV bar (UTC)."""

    ts: pd.Timestamp
    open: float
    high: float
    low: float
    close: float
    volume: float | None = None

    def to_series_row(self) -> dict:
        row = {
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
        }
        if self.volume is not None:
            row["volume"] = self.volume
        return row


def candle_to_timestamp(ts: datetime | pd.Timestamp) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    if t.tz is None:
        return t.tz_localize("UTC")
    return t.tz_convert("UTC")

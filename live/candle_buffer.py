"""Rolling OHLCV buffer for live + replay ingestion."""

from __future__ import annotations

from collections import deque

import pandas as pd

from live.live_types import Candle, candle_to_timestamp


class CandleBuffer:
    """
    Fixed-capacity deque of candles; exposes a UTC-indexed DataFrame for strategy code.

    ``min_bars`` should cover Asian build + MSS forward window (e.g. 2000+ hours).
    """

    def __init__(self, max_bars: int = 5000) -> None:
        self.max_bars = int(max_bars)
        self._dq: deque[Candle] = deque(maxlen=self.max_bars)

    def __len__(self) -> int:
        return len(self._dq)

    def append(self, c: Candle) -> None:
        self._dq.append(c)

    def last_ts(self) -> pd.Timestamp | None:
        if not self._dq:
            return None
        return candle_to_timestamp(self._dq[-1].ts)

    def to_dataframe(self) -> pd.DataFrame:
        if not self._dq:
            return pd.DataFrame(
                columns=["open", "high", "low", "close"],
                index=pd.DatetimeIndex([], tz="UTC", name="datetime"),
            )
        rows = [c.to_series_row() for c in self._dq]
        idx = pd.DatetimeIndex([candle_to_timestamp(c.ts) for c in self._dq], name="datetime")
        return pd.DataFrame(rows, index=pd.DatetimeIndex(idx, tz="UTC"))

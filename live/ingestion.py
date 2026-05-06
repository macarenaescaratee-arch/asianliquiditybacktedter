"""Live candle ingestion: abstract source + CSV replay + exchange stub."""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from pathlib import Path
from typing import AsyncIterator, Iterator

import pandas as pd

from live.live_types import Candle, candle_to_timestamp


class BarSource(ABC):
    """Push or pull closed bars (one bar at a time)."""

    @abstractmethod
    def symbol(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def timeframe(self) -> str:
        """Human label e.g. ``1h``."""
        raise NotImplementedError


class CsvReplaySource(BarSource):
    """
    Deterministic replay from project OHLCV CSV (same schema as ``data.loader``).

    Iterates closed bars in time order for dry-runs.
    """

    def __init__(
        self,
        csv_path: str | Path,
        symbol: str = "EURUSD",
        timeframe: str = "1h",
        *,
        datetime_column: str = "datetime",
        tail_rows: int | None = None,
    ) -> None:
        from data.loader import load_ohlcv_csv

        self._path = Path(csv_path)
        self._symbol = symbol
        self._tf = timeframe
        df = load_ohlcv_csv(self._path, datetime_column=datetime_column)
        if tail_rows is not None and tail_rows > 0:
            df = df.iloc[-int(tail_rows) :].copy()
        self._df = df
        self._i = 0

    def symbol(self) -> str:
        return self._symbol

    def timeframe(self) -> str:
        return self._tf

    def __iter__(self) -> Iterator[Candle]:
        return self

    def __next__(self) -> Candle:
        if self._i >= len(self._df):
            raise StopIteration
        row = self._df.iloc[self._i]
        ts = self._df.index[self._i]
        self._i += 1
        vol = float(row["volume"]) if "volume" in self._df.columns else None
        return Candle(
            ts=candle_to_timestamp(ts),
            open=float(row["open"]),
            high=float(row["high"]),
            low=float(row["low"]),
            close=float(row["close"]),
            volume=vol,
        )

    def reset(self) -> None:
        self._i = 0


class WebSocketBarSourceStub(BarSource):
    """
    Placeholder for real-time feed (Binance, Oanda, FXCM, MT5 bridge, etc.).

    Subclass and implement ``connect`` / bar normalization; this stub documents the contract.
    """

    def __init__(self, symbol: str = "EURUSD", timeframe: str = "1h") -> None:
        self._symbol = symbol
        self._tf = timeframe

    def symbol(self) -> str:
        return self._symbol

    def timeframe(self) -> str:
        return self._tf

    async def stream(self) -> AsyncIterator[Candle]:
        """Override in production: parse JSON frames into ``Candle``."""
        if False:
            yield Candle(
                ts=pd.Timestamp.now(tz="UTC"),
                open=0.0,
                high=0.0,
                low=0.0,
                close=0.0,
            )
        await asyncio.sleep(0)
        return

"""Orchestrates ingestion → buffer → signals → quant trade management."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from backtester.eurusd_phase4_execution import compute_atr, phase4_eurusd_institutional_baseline
from live.broker.base import BrokerClient
from live.broker.paper_broker import PaperBrokerClient
from live.candle_buffer import CandleBuffer
from live.config import MIN_OHLC_BARS_WARMUP
from live.ingestion import BarSource, CsvReplaySource
from live.signal_engine import InstitutionalEURUSDSignalEngine
from live.trade_logger import TradeLogger
from live.trade_manager import QuantTradeManager
from live.live_types import Candle
from strategy.asian_liquidity_mss import default_liquidity_mss_config


@dataclass
class BotRuntimeSnapshot:
    last_processed_ts: str | None
    fired_keys: list[tuple[str, str]]
    manager: dict
    metrics: dict[str, int]


class LiveEURUSDBot:
    """
    Production-oriented skeleton: single-threaded closed-bar loop.

    Swap ``CsvReplaySource`` for a real ``WebSocketBarSourceStub`` subclass when going live.
    """

    def __init__(
        self,
        *,
        bar_source: BarSource,
        min_bars: int = MIN_OHLC_BARS_WARMUP,
        buffer_capacity: int = 6000,
        log_path: Path | None = None,
        default_units: float = 1.0,
        broker: BrokerClient | None = None,
    ) -> None:
        self.source = bar_source
        self.buffer = CandleBuffer(max_bars=buffer_capacity)
        cfg = default_liquidity_mss_config()
        self.engine = InstitutionalEURUSDSignalEngine(cfg, min_bars=min_bars)
        self.logger = TradeLogger(path=log_path)
        self.broker = broker or PaperBrokerClient(default_units=default_units)
        self.last_processed_ts: pd.Timestamp | None = None
        self.metrics: dict[str, int] = {
            "duplicate_candles": 0,
            "processed_candles": 0,
            "warmup_candles": 0,
        }
        self.manager = QuantTradeManager(
            phase4_eurusd_institutional_baseline(),
            self.broker,
            self.logger,
        )

    def _append_unique_candle(self, candle: Candle) -> bool:
        ts = pd.Timestamp(candle.ts)
        if self.last_processed_ts is not None and ts <= self.last_processed_ts:
            self.metrics["duplicate_candles"] += 1
            self.logger.log(
                "duplicate_candle_skip",
                {"bar_ts": str(ts), "last_processed_ts": str(self.last_processed_ts)},
            )
            return False
        self.buffer.append(candle)
        self.last_processed_ts = ts
        self.metrics["processed_candles"] += 1
        return True

    def on_warmup_bar(self, candle: Candle) -> None:
        if self._append_unique_candle(candle):
            self.metrics["warmup_candles"] += 1

    def on_closed_bar(self, candle: Candle, *, allow_trading: bool = True) -> None:
        if not self._append_unique_candle(candle):
            return
        df = self.buffer.to_dataframe()
        if len(df) < self.engine.min_bars:
            return
        if not allow_trading:
            return
        atr = compute_atr(df)
        idx = len(df) - 1
        new_setups = self.engine.scan_new_entries(df)
        snap = self.engine.last_session_snapshot(df)
        if snap is not None and "candidate" in snap.tag.lower():
            self.logger.log(
                "asian_snapshot",
                {
                    "bar_ts": str(candle.ts),
                    "session_date": str(snap.session_date),
                    "asian_high": snap.asian_high,
                    "asian_low": snap.asian_low,
                    "tag": snap.tag,
                    "sweep_side": snap.sweep_side,
                    "mss_direction": snap.mss_direction,
                },
            )
        self.manager.on_candle_closed(df, atr, idx, new_setups)

    def snapshot_runtime(self) -> BotRuntimeSnapshot:
        return BotRuntimeSnapshot(
            last_processed_ts=(
                pd.Timestamp(self.last_processed_ts).isoformat()
                if self.last_processed_ts is not None
                else None
            ),
            fired_keys=self.engine.snapshot_fired_keys(),
            manager=self.manager.snapshot(),
            metrics=dict(self.metrics),
        )

    def restore_runtime(self, payload: dict) -> bool:
        ok = True
        ts = payload.get("last_processed_ts")
        self.last_processed_ts = pd.Timestamp(ts) if ts else None
        self.engine.restore_fired_keys(payload.get("fired_keys") or [])
        ok = self.manager.restore_from_snapshot(payload.get("manager") or {}) and ok
        m = payload.get("metrics") or {}
        if isinstance(m, dict):
            for k, v in m.items():
                if k in self.metrics:
                    self.metrics[k] = int(v)
        return ok

    def run_replay_to_end(self) -> None:
        for c in self.source:
            self.on_closed_bar(c)


def default_replay_bot(csv_path: str | Path) -> LiveEURUSDBot:
    return LiveEURUSDBot(bar_source=CsvReplaySource(csv_path, symbol="EURUSD", timeframe="1h"))

"""Asian sweep + MSS + Phase 3 EURUSD institutional filters → actionable setups."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd

from backtester.asian_mss_execution import TradeSetup, build_trade_setups
from strategy.asian_liquidity_mss import LiquidityMSSConfig, detect_asian_liquidity_mss


@dataclass(slots=True)
class AsianSessionSnapshot:
    session_date: date
    asian_high: float
    asian_low: float
    sweep_side: str
    mss_direction: str
    tag: str


class InstitutionalEURUSDSignalEngine:
    """
    Batch-aligned signal engine: on each closed bar, re-runs detector on the full buffer.

    Emits ``TradeSetup`` only when ``entry_ts`` equals the **last** bar (just closed).
    Production note: replace with incremental scan for latency if needed.
    """

    def __init__(
        self,
        cfg: LiquidityMSSConfig,
        *,
        symbol: str = "EURUSD",
        min_bars: int = 900,
        institutional: bool = True,
    ) -> None:
        self.cfg = cfg
        self.symbol = symbol
        self.min_bars = min_bars
        self.institutional = institutional
        self._fired_keys: set[tuple[str, str]] = set()

    def reset_fired_keys(self) -> None:
        """Call before a new replay / session if the same history is fed again."""
        self._fired_keys.clear()

    def snapshot_fired_keys(self) -> list[tuple[str, str]]:
        return sorted(self._fired_keys)

    def restore_fired_keys(self, keys: list[tuple[str, str]] | list[list[str]]) -> None:
        restored: set[tuple[str, str]] = set()
        for k in keys:
            if isinstance(k, (list, tuple)) and len(k) == 2:
                restored.add((str(k[0]), str(k[1])))
        self._fired_keys = restored

    def last_session_snapshot(self, ohlcv: pd.DataFrame) -> AsianSessionSnapshot | None:
        if len(ohlcv) < self.min_bars:
            return None
        daily = detect_asian_liquidity_mss(
            ohlcv,
            self.cfg,
            symbol=self.symbol,
            eurusd_institutional=self.institutional,
        )
        if daily.empty:
            return None
        row = daily.iloc[-1]
        return AsianSessionSnapshot(
            session_date=row["session_date"],
            asian_high=float(row["asian_high"]),
            asian_low=float(row["asian_low"]),
            sweep_side=str(row["sweep_side"]),
            mss_direction=str(row["mss_direction"]),
            tag=str(row["tag"]),
        )

    def scan_new_entries(self, ohlcv: pd.DataFrame) -> list[TradeSetup]:
        if len(ohlcv) < self.min_bars:
            return []
        daily = detect_asian_liquidity_mss(
            ohlcv,
            self.cfg,
            symbol=self.symbol,
            eurusd_institutional=self.institutional,
        )
        setups = build_trade_setups(self.symbol, ohlcv, self.cfg, daily)
        last_ts = ohlcv.index[-1]
        out: list[TradeSetup] = []
        for s in setups:
            if s.entry_ts != last_ts:
                continue
            key = (pd.Timestamp(s.entry_ts).isoformat(), s.direction)
            if key in self._fired_keys:
                continue
            self._fired_keys.add(key)
            out.append(s)
        return out

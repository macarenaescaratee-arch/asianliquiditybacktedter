"""Executable order plan: bridge ``TradeSetup`` + Phase 4 to broker-ready prices."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd

from backtester.asian_mss_execution import TradeSetup
from backtester.eurusd_phase4_execution import (
    ExecutionVariant,
    _adj_entry,
    compute_sl_tp_risk,
    entry_bar_index,
)


@dataclass
class LiveExecutionPlan:
    """Prices and metadata for order placement (after spread-adjusted entry model)."""

    symbol: str
    session_date: date
    direction: str
    entry_ts: pd.Timestamp
    sweep_ts: pd.Timestamp
    confirm_ts: pd.Timestamp
    asian_high: float
    asian_low: float
    entry_price: float
    stop_price: float
    take_profit_price: float
    risk_points: float
    risk_pips: float
    rr_target: float
    spread_pips: float
    max_loss_cap_pips: float

    @classmethod
    def from_setup(
        cls,
        setup: TradeSetup,
        ohlcv: pd.DataFrame,
        atr: pd.Series,
        v: ExecutionVariant,
    ) -> LiveExecutionPlan:
        i0 = entry_bar_index(ohlcv, setup.entry_ts)
        entry = _adj_entry(float(setup.entry), setup.direction, v.spread_pips)
        sl, tp, risk = compute_sl_tp_risk(setup, ohlcv, atr, entry, i0, v)
        pip = setup.pip_or_point_size
        risk_pips = risk / pip if pip > 0 else risk
        if tp is not None and risk > 1e-12:
            if setup.direction == "long":
                rr_eff = (float(tp) - float(entry)) / risk
            else:
                rr_eff = (float(entry) - float(tp)) / risk
        else:
            rr_eff = float(v.tp_rr)
        return cls(
            symbol=setup.symbol,
            session_date=setup.session_date,
            direction=setup.direction,
            entry_ts=setup.entry_ts,
            sweep_ts=setup.sweep_ts,
            confirm_ts=setup.confirm_ts,
            asian_high=setup.asian_high,
            asian_low=setup.asian_low,
            entry_price=float(entry),
            stop_price=float(sl),
            take_profit_price=float(tp) if tp is not None else float("nan"),
            risk_points=float(risk),
            risk_pips=float(risk_pips),
            rr_target=float(rr_eff),
            spread_pips=float(v.spread_pips),
            max_loss_cap_pips=float(v.max_loss_cap_pips),
        )

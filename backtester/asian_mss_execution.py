"""
Asian Liquidity Sweep + MSS: trade plan extraction and bar-based simulation.

Converts validated MSS setups into entry / stop / take-profit (minimum 1:2 RR),
direction, and risk in price points (and pip-equivalent for FX).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Literal

import numpy as np
import pandas as pd

from strategy.asian_liquidity_mss import (
    LiquidityMSSConfig,
    annotate_tokyo_calendar,
    build_london_ny_killzone_mask,
    fractal_pivot_high,
    fractal_pivot_low,
    sweep_session_mask,
    _first_liquidity_sweep,
    _is_bearish_displacement,
    _is_bullish_displacement,
    _segment_range_ref,
)

Direction = Literal["long", "short"]
Outcome = Literal["win", "loss", "timeout"]


@dataclass(slots=True)
class TradeSetup:
    symbol: str
    session_date: date
    direction: Direction
    sweep_ts: pd.Timestamp
    confirm_ts: pd.Timestamp
    entry_ts: pd.Timestamp
    entry: float
    stop: float
    take_profit: float
    rr_target: float
    risk_points: float
    pip_or_point_size: float
    risk_in_pips: float
    asian_high: float
    asian_low: float


@dataclass(slots=True)
class SimulatedTrade:
    setup: TradeSetup
    outcome: Outcome
    realized_rr: float
    exit_ts: pd.Timestamp | None
    exit_price: float | None


def _pip_size(symbol: str) -> float:
    """Price increment used for pip/point reporting (pragmatic per asset class)."""
    s = symbol.upper()
    if s in {"EURUSD", "GBPUSD"}:
        return 0.0001
    if s == "XAUUSD":
        return 0.01
    if s == "NAS100":
        return 0.01
    return 0.0001


def _sl_buffer(symbol: str) -> float:
    return _pip_size(symbol) * 3.0


def _bullish_confirm_ts(
    seg_after: pd.DataFrame,
    cfg: LiquidityMSSConfig,
    sweep_ts: pd.Timestamp,
) -> pd.Timestamp | None:
    """First MSS confirmation timestamp (mirrors ``_bullish_mss_after_sweep_down``)."""
    if seg_after.empty:
        return None
    ref_range = _segment_range_ref(seg_after)
    disp_i = None
    for ts, row in seg_after.iterrows():
        if ts <= sweep_ts:
            continue
        if _is_bullish_displacement(row, ref_range, cfg.displacement_body_ratio, cfg.displacement_min_range_ratio):
            disp_i = ts
            break
    if disp_i is None:
        return None
    pos = int(seg_after.index.get_indexer([disp_i])[0])
    if pos < 0:
        return None
    tail = seg_after.iloc[pos:]
    lows = tail["low"]
    highs = tail["high"]
    fh = fractal_pivot_high(lows, highs, cfg.fractal_left, cfg.fractal_right)
    post = tail[fh]
    if post.empty:
        anchor_high = float(tail["high"].iloc[0])
        for ts, row in tail.iloc[1:].iterrows():
            if float(row["close"]) > anchor_high:
                return ts
        return None
    last_anchor = float(post["high"].iloc[-1])
    after_last = tail.loc[tail.index > post.index[-1]]
    for ts, row in after_last.iterrows():
        if float(row["close"]) > last_anchor:
            return ts
    return None


def _bearish_confirm_ts(
    seg_after: pd.DataFrame,
    cfg: LiquidityMSSConfig,
    sweep_ts: pd.Timestamp,
) -> pd.Timestamp | None:
    """First MSS confirmation timestamp (mirrors ``_bearish_mss_after_sweep_up``)."""
    if seg_after.empty:
        return None
    ref_range = _segment_range_ref(seg_after)
    disp_i = None
    for ts, row in seg_after.iterrows():
        if ts <= sweep_ts:
            continue
        if _is_bearish_displacement(row, ref_range, cfg.displacement_body_ratio, cfg.displacement_min_range_ratio):
            disp_i = ts
            break
    if disp_i is None:
        return None
    pos = int(seg_after.index.get_indexer([disp_i])[0])
    if pos < 0:
        return None
    tail = seg_after.iloc[pos:]
    lows = tail["low"]
    highs = tail["high"]
    fl = fractal_pivot_low(lows, highs, cfg.fractal_left, cfg.fractal_right)
    post = tail[fl]
    if post.empty:
        anchor_low = float(tail["low"].iloc[0])
        for ts, row in tail.iloc[1:].iterrows():
            if float(row["close"]) < anchor_low:
                return ts
        return None
    last_anchor = float(post["low"].iloc[-1])
    after_last = tail.loc[tail.index > post.index[-1]]
    for ts, row in after_last.iterrows():
        if float(row["close"]) < last_anchor:
            return ts
    return None


def _seg_after(df: pd.DataFrame, sweep_ts: pd.Timestamp, cfg: LiquidityMSSConfig) -> pd.DataFrame:
    cols = ["open", "high", "low", "close"]
    end_scan = sweep_ts + pd.Timedelta(hours=cfg.mss_forward_hours)
    return df.loc[(df.index > sweep_ts) & (df.index <= end_scan), cols].copy()


def build_trade_setups(
    symbol: str,
    ohlcv: pd.DataFrame,
    cfg: LiquidityMSSConfig,
    daily_signals: pd.DataFrame,
) -> list[TradeSetup]:
    """
    Build executable trade plans for rows tagged as MSS setup candidates.

    ``daily_signals`` must be the output of ``detect_asian_liquidity_mss`` aligned to ``ohlcv``.
    """
    df = annotate_tokyo_calendar(ohlcv)
    killzone = build_london_ny_killzone_mask(df.index)
    cols = ["open", "high", "low", "close"]
    base = df.loc[:, cols]
    pip = _pip_size(symbol)
    buf = _sl_buffer(symbol)
    setups: list[TradeSetup] = []

    for _, row in daily_signals.iterrows():
        if "candidate" not in str(row.get("tag", "")):
            continue
        D = row["session_date"]
        ah, al = float(row["asian_high"]), float(row["asian_low"])
        sweep_side = str(row["sweep_side"])
        mss_dir = str(row["mss_direction"])
        sm = sweep_session_mask(df, D, killzone)
        seg = df.loc[sm, cols].copy()
        ss, sweep_ts = _first_liquidity_sweep(seg, ah, al)
        if sweep_ts is None or ss == "none":
            continue
        seg_after = _seg_after(base, sweep_ts, cfg)

        if mss_dir == "bullish" and sweep_side == "below":
            confirm_ts = _bullish_confirm_ts(seg_after, cfg, sweep_ts)
            direction: Direction = "long"
        elif mss_dir == "bearish" and sweep_side == "above":
            confirm_ts = _bearish_confirm_ts(seg_after, cfg, sweep_ts)
            direction = "short"
        else:
            continue
        if confirm_ts is None:
            continue

        after_confirm = base.loc[base.index > confirm_ts]
        if after_confirm.empty:
            continue
        entry_row = after_confirm.iloc[0]
        entry_ts = after_confirm.index[0]
        entry = float(entry_row["open"])

        sweep_low = float(base.loc[sweep_ts, "low"])
        sweep_high = float(base.loc[sweep_ts, "high"])

        if direction == "long":
            sl = min(al, sweep_low) - buf
            if sl >= entry:
                sl = entry - buf * 2
            risk = entry - sl
            if risk <= 0:
                continue
            tp = entry + 2.0 * risk
        else:
            sl = max(ah, sweep_high) + buf
            if sl <= entry:
                sl = entry + buf * 2
            risk = sl - entry
            if risk <= 0:
                continue
            tp = entry - 2.0 * risk

        rr_target = 2.0
        risk_pips = risk / pip if pip > 0 else risk

        setups.append(
            TradeSetup(
                symbol=symbol,
                session_date=D if isinstance(D, date) else pd.Timestamp(D).date(),
                direction=direction,
                sweep_ts=sweep_ts,
                confirm_ts=confirm_ts,
                entry_ts=entry_ts,
                entry=entry,
                stop=sl,
                take_profit=tp,
                rr_target=rr_target,
                risk_points=risk,
                pip_or_point_size=pip,
                risk_in_pips=risk_pips,
                asian_high=ah,
                asian_low=al,
            )
        )

    return setups


def _simulate_one(ohlcv: pd.DataFrame, t: TradeSetup, max_bars: int = 200) -> SimulatedTrade:
    """Bar path: pessimistic same-bar rule if both TP and SL touched (stop first for longs at TP level...)."""
    loc = ohlcv.index.get_loc(t.entry_ts)
    if isinstance(loc, slice):
        start = int(loc.start) if loc.start is not None else 0
    elif isinstance(loc, np.ndarray):
        start = int(loc.min()) if len(loc) else 0
    elif isinstance(loc, (list, tuple)):
        start = int(min(loc))
    else:
        start = int(loc)
    outcome: Outcome = "timeout"
    realized_rr = 0.0
    exit_ts: pd.Timestamp | None = None
    exit_px: float | None = None

    for i in range(start, min(start + max_bars, len(ohlcv))):
        bar = ohlcv.iloc[i]
        hi, lo, cl = float(bar["high"]), float(bar["low"]), float(bar["close"])
        ts = ohlcv.index[i]

        if t.direction == "long":
            hit_sl = lo <= t.stop
            hit_tp = hi >= t.take_profit
            if hit_sl and hit_tp:
                outcome = "loss"
                realized_rr = -1.0
                exit_ts, exit_px = ts, t.stop
                break
            if hit_sl:
                outcome = "loss"
                realized_rr = -1.0
                exit_ts, exit_px = ts, t.stop
                break
            if hit_tp:
                outcome = "win"
                realized_rr = t.rr_target
                exit_ts, exit_px = ts, t.take_profit
                break
        else:
            hit_sl = hi >= t.stop
            hit_tp = lo <= t.take_profit
            if hit_sl and hit_tp:
                outcome = "loss"
                realized_rr = -1.0
                exit_ts, exit_px = ts, t.stop
                break
            if hit_sl:
                outcome = "loss"
                realized_rr = -1.0
                exit_ts, exit_px = ts, t.stop
                break
            if hit_tp:
                outcome = "win"
                realized_rr = t.rr_target
                exit_ts, exit_px = ts, t.take_profit
                break

    if outcome == "timeout":
        last = ohlcv.iloc[min(start + max_bars - 1, len(ohlcv) - 1)]
        exit_ts = ohlcv.index[min(start + max_bars - 1, len(ohlcv) - 1)]
        exit_px = float(last["close"])
        r = (exit_px - t.entry) / t.risk_points if t.direction == "long" else (t.entry - exit_px) / t.risk_points
        realized_rr = float(r)

    return SimulatedTrade(
        setup=t,
        outcome=outcome,
        realized_rr=realized_rr,
        exit_ts=exit_ts,
        exit_price=exit_px,
    )


def simulate_all(ohlcv: pd.DataFrame, setups: list[TradeSetup]) -> list[SimulatedTrade]:
    return [_simulate_one(ohlcv, s) for s in setups]


def summarize_by_symbol(results: list[SimulatedTrade]) -> pd.DataFrame:
    if not results:
        return pd.DataFrame()
    df = pd.DataFrame(
        {
            "symbol": [r.setup.symbol for r in results],
            "outcome": [r.outcome for r in results],
            "rr": [r.realized_rr for r in results],
        }
    )
    agg = []
    for sym, g in df.groupby("symbol"):
        n = len(g)
        wins = int((g["outcome"] == "win").sum())
        losses = int((g["outcome"] == "loss").sum())
        to = int((g["outcome"] == "timeout").sum())
        wr = wins / n if n else 0.0
        avg_rr = float(g["rr"].mean()) if n else 0.0
        exp = float(g["rr"].mean()) if n else 0.0
        agg.append(
            {
                "symbol": sym,
                "trades": n,
                "wins": wins,
                "losses": losses,
                "timeouts": to,
                "win_rate": wr,
                "avg_rr": avg_rr,
                "expectancy_r": exp,
            }
        )
    out = pd.DataFrame(agg)
    if not out.empty:
        best = out.loc[out["expectancy_r"].idxmax(), "symbol"] if len(out) > 1 else out["symbol"].iloc[0]
        worst = out.loc[out["expectancy_r"].idxmin(), "symbol"] if len(out) > 1 else out["symbol"].iloc[0]
        out.attrs["best_asset"] = best
        out.attrs["worst_asset"] = worst
    return out

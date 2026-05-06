"""
Phase 4 — EURUSD institutional execution / trade-management simulation.

Does not alter signal detection or Phase 3 filters; only post-entry paths.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from backtester.asian_mss_execution import TradeSetup

PIP = 0.0001


def entry_bar_index(ohlcv: pd.DataFrame, entry_ts: pd.Timestamp) -> int:
    loc = ohlcv.index.get_loc(entry_ts)
    if isinstance(loc, slice):
        return int(loc.start) if loc.start is not None else 0
    if isinstance(loc, np.ndarray):
        return int(loc.min()) if len(loc) else 0
    if isinstance(loc, (list, tuple)):
        return int(min(loc))
    return int(loc)


def compute_atr(ohlcv: pd.DataFrame, period: int = 14) -> pd.Series:
    h, l, c = ohlcv["high"], ohlcv["low"], ohlcv["close"]
    pc = c.shift(1)
    tr = pd.concat([(h - l).abs(), (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.rolling(period, min_periods=5).mean()


def atr_value(atr: pd.Series, ts: pd.Timestamp) -> float:
    s = atr.reindex([ts]).ffill().bfill()
    if len(s) == 0 or pd.isna(s.iloc[0]):
        return float("nan")
    return float(s.iloc[0])


@dataclass(slots=True)
class ExecutionVariant:
    name: str
    sl_mode: Literal["structural", "fixed_pip", "atr", "hybrid_struct_atr", "structural_capped"]
    tp_mode: Literal["fixed_rr", "opp_liquidity", "session_close_ny", "trail_structure", "partial_1r_runner"]
    tp_rr: float = 2.0
    sl_fixed_pips: float = 15.0
    sl_atr_mult: float = 1.25
    max_loss_cap_pips: float = 35.0
    be_after_r: float | None = None
    partial_pct_at_1r: float | None = None
    runner_tp_rr: float = 2.5
    time_stop_hours: float | None = None
    min_expansion_r: float = 0.22
    kill_ny_utc_hour: int | None = None
    spread_pips: float = 0.35
    """Adverse slippage per side in pips; round-trip cost applied in R at end of sim."""
    slippage_pips: float = 0.0
    trail_atr_mult: float = 0.4
    london_exit_utc_hour: int = 16
    max_bars: int = 240


@dataclass(slots=True)
class Phase4SimResult:
    realized_r: float
    outcome: str
    exit_ts: pd.Timestamp | None
    initial_risk_price: float
    mfe_r: float = 0.0


@dataclass(slots=True)
class Phase4LiveState:
    """Mutable intrabar state for causal live stepping (mirrors ``simulate_variant`` rules)."""

    setup: TradeSetup
    v: ExecutionVariant
    start_i: int
    entry: float
    risk0: float
    sl_live: float
    tp_live: float | None
    pos: float
    total_r: float
    be_armed: bool
    partial_done: bool
    one_r_px: float
    mfe_r: float
    closed: bool = False


def phase4_live_init(ohlcv: pd.DataFrame, setup: TradeSetup, v: ExecutionVariant, atr: pd.Series) -> Phase4LiveState:
    start = entry_bar_index(ohlcv, setup.entry_ts)
    entry = _adj_entry(float(setup.entry), setup.direction, v.spread_pips)
    sl, tp_live, risk0 = compute_sl_tp_risk(setup, ohlcv, atr, entry, start, v)
    if risk0 <= 1e-12:
        return Phase4LiveState(
            setup=setup,
            v=v,
            start_i=start,
            entry=entry,
            risk0=risk0,
            sl_live=sl,
            tp_live=tp_live,
            pos=0.0,
            total_r=0.0,
            be_armed=False,
            partial_done=False,
            one_r_px=entry,
            mfe_r=0.0,
            closed=True,
        )
    one_r = entry + risk0 if setup.direction == "long" else entry - risk0
    return Phase4LiveState(
        setup=setup,
        v=v,
        start_i=start,
        entry=entry,
        risk0=risk0,
        sl_live=sl,
        tp_live=tp_live,
        pos=1.0,
        total_r=0.0,
        be_armed=False,
        partial_done=False,
        one_r_px=one_r,
        mfe_r=0.0,
        closed=False,
    )


def phase4_live_step(
    state: Phase4LiveState,
    ohlcv: pd.DataFrame,
    atr: pd.Series,
    bar_i: int,
) -> tuple[Phase4LiveState, Phase4SimResult | None]:
    """
    Process one closed bar (index ``bar_i`` in ``ohlcv``). Returns updated state and a result if the trade closed.
    """
    if state.closed or state.risk0 <= 1e-12:
        return state, None
    v = state.v
    setup = state.setup
    if bar_i < state.start_i:
        return state, None

    ts = ohlcv.index[bar_i]
    bar = ohlcv.iloc[bar_i]
    hi, lo, cl = float(bar["high"]), float(bar["low"]), float(bar["close"])

    if setup.direction == "long":
        state.mfe_r = max(state.mfe_r, (hi - state.entry) / state.risk0)
    else:
        state.mfe_r = max(state.mfe_r, (state.entry - lo) / state.risk0)

    if v.be_after_r is not None and not state.be_armed:
        if setup.direction == "long" and hi >= state.entry + v.be_after_r * state.risk0:
            state.sl_live = max(state.sl_live, state.entry - 0.5 * PIP)
            state.be_armed = True
        elif setup.direction == "short" and lo <= state.entry - v.be_after_r * state.risk0:
            state.sl_live = min(state.sl_live, state.entry + 0.5 * PIP)
            state.be_armed = True

    if (
        v.tp_mode == "partial_1r_runner"
        and v.partial_pct_at_1r
        and not state.partial_done
        and state.pos > 0.99
    ):
        hit_1r = hi >= state.one_r_px if setup.direction == "long" else lo <= state.one_r_px
        if hit_1r:
            frac = float(v.partial_pct_at_1r)
            state.total_r += frac * 1.0 * state.pos
            state.pos *= 1.0 - frac
            state.partial_done = True
            state.sl_live = state.entry - 0.5 * PIP if setup.direction == "long" else state.entry + 0.5 * PIP
            if setup.direction == "long":
                state.tp_live = state.entry + v.runner_tp_rr * state.risk0
            else:
                state.tp_live = state.entry - v.runner_tp_rr * state.risk0

    if v.tp_mode == "trail_structure" and bar_i > state.start_i:
        av = atr_value(atr, ts)
        if not np.isnan(av):
            eps = 0.5 * PIP
            if setup.direction == "long":
                trail = hi - v.trail_atr_mult * av
                if not state.be_armed:
                    trail = min(trail, state.entry - eps)
                state.sl_live = max(state.sl_live, trail)
            else:
                trail = lo + v.trail_atr_mult * av
                if not state.be_armed:
                    trail = max(trail, state.entry + eps)
                state.sl_live = min(state.sl_live, trail)

    if v.tp_mode == "session_close_ny" and ts.hour >= v.london_exit_utc_hour:
        r_rem = ((cl - state.entry) / state.risk0) * state.pos if setup.direction == "long" else (
            (state.entry - cl) / state.risk0
        ) * state.pos
        total = state.total_r + r_rem
        if v.slippage_pips > 0:
            total -= (2.0 * v.slippage_pips * PIP) / state.risk0
        res = Phase4SimResult(float(total), "session_exit", ts, state.risk0, float(state.mfe_r))
        state.closed = True
        return state, res

    if v.kill_ny_utc_hour is not None:
        if ts.date() == ohlcv.index[state.start_i].date() and ts.hour >= v.kill_ny_utc_hour:
            r_rem = ((cl - state.entry) / state.risk0) * state.pos if setup.direction == "long" else (
                (state.entry - cl) / state.risk0
            ) * state.pos
            total = state.total_r + r_rem
            if v.slippage_pips > 0:
                total -= (2.0 * v.slippage_pips * PIP) / state.risk0
            res = Phase4SimResult(float(total), "ny_kill", ts, state.risk0, float(state.mfe_r))
            state.closed = True
            return state, res

    if v.time_stop_hours is not None:
        hrs = (ts - ohlcv.index[state.start_i]).total_seconds() / 3600.0
        if hrs >= v.time_stop_hours and state.mfe_r < v.min_expansion_r:
            r_rem = ((cl - state.entry) / state.risk0) * state.pos if setup.direction == "long" else (
                (state.entry - cl) / state.risk0
            ) * state.pos
            total = state.total_r + r_rem
            if v.slippage_pips > 0:
                total -= (2.0 * v.slippage_pips * PIP) / state.risk0
            res = Phase4SimResult(float(total), "time_stop", ts, state.risk0, float(state.mfe_r))
            state.closed = True
            return state, res

    tp_live = state.tp_live
    if setup.direction == "long":
        hit_sl = lo <= state.sl_live
        tp_active = tp_live if state.pos > 1e-6 else None
        hit_tp = tp_active is not None and hi >= tp_active
        if hit_sl and hit_tp:
            total = state.total_r - 1.0 * state.pos
            if v.slippage_pips > 0:
                total -= (2.0 * v.slippage_pips * PIP) / state.risk0
            res = Phase4SimResult(float(total), "loss", ts, state.risk0, float(state.mfe_r))
            state.closed = True
            return state, res
        if hit_sl:
            total = state.total_r - 1.0 * state.pos
            if v.slippage_pips > 0:
                total -= (2.0 * v.slippage_pips * PIP) / state.risk0
            res = Phase4SimResult(float(total), "loss", ts, state.risk0, float(state.mfe_r))
            state.closed = True
            return state, res
        if hit_tp:
            rr_hit = (tp_live - state.entry) / state.risk0
            total = state.total_r + rr_hit * state.pos
            if v.slippage_pips > 0:
                total -= (2.0 * v.slippage_pips * PIP) / state.risk0
            res = Phase4SimResult(float(total), "win", ts, state.risk0, float(state.mfe_r))
            state.closed = True
            return state, res
    else:
        hit_sl = hi >= state.sl_live
        hit_tp = tp_live is not None and lo <= tp_live and state.pos > 1e-6
        if hit_sl and hit_tp:
            total = state.total_r - 1.0 * state.pos
            if v.slippage_pips > 0:
                total -= (2.0 * v.slippage_pips * PIP) / state.risk0
            res = Phase4SimResult(float(total), "loss", ts, state.risk0, float(state.mfe_r))
            state.closed = True
            return state, res
        if hit_sl:
            total = state.total_r - 1.0 * state.pos
            if v.slippage_pips > 0:
                total -= (2.0 * v.slippage_pips * PIP) / state.risk0
            res = Phase4SimResult(float(total), "loss", ts, state.risk0, float(state.mfe_r))
            state.closed = True
            return state, res
        if hit_tp:
            rr_hit = (state.entry - tp_live) / state.risk0
            total = state.total_r + rr_hit * state.pos
            if v.slippage_pips > 0:
                total -= (2.0 * v.slippage_pips * PIP) / state.risk0
            res = Phase4SimResult(float(total), "win", ts, state.risk0, float(state.mfe_r))
            state.closed = True
            return state, res

    last_allowed_i = min(state.start_i + v.max_bars, len(ohlcv)) - 1
    if bar_i >= last_allowed_i:
        cl = float(ohlcv.iloc[bar_i]["close"])
        r_rem = ((cl - state.entry) / state.risk0) * state.pos if setup.direction == "long" else (
            (state.entry - cl) / state.risk0
        ) * state.pos
        total = state.total_r + r_rem
        if v.slippage_pips > 0:
            total -= (2.0 * v.slippage_pips * PIP) / state.risk0
        res = Phase4SimResult(float(total), "timeout", ts, state.risk0, float(state.mfe_r))
        state.closed = True
        return state, res

    return state, None


def _cap_risk(entry: float, sl: float, direction: str, cap_pips: float) -> tuple[float, float]:
    """Return (sl, risk) after optional max-width cap from entry."""
    if direction == "long":
        risk = entry - sl
        if risk > cap_pips * PIP:
            sl = entry - cap_pips * PIP
            risk = entry - sl
        return sl, max(risk, 1e-12)
    risk = sl - entry
    if risk > cap_pips * PIP:
        sl = entry + cap_pips * PIP
        risk = sl - entry
    return sl, max(risk, 1e-12)


def compute_sl_tp_risk(
    setup: TradeSetup,
    ohlcv: pd.DataFrame,
    atr: pd.Series,
    entry: float,
    entry_i: int,
    v: ExecutionVariant,
) -> tuple[float, float | None, float]:
    buf = 3.0 * PIP
    ah, al = setup.asian_high, setup.asian_low
    sweep_low = float(ohlcv.loc[setup.sweep_ts, "low"])
    sweep_high = float(ohlcv.loc[setup.sweep_ts, "high"])
    av = atr_value(atr, ohlcv.index[entry_i])

    if setup.direction == "long":
        sl_struct = min(al, sweep_low) - buf
        if sl_struct >= entry:
            sl_struct = entry - buf * 2

        if v.sl_mode == "structural":
            sl = sl_struct
        elif v.sl_mode == "fixed_pip":
            sl = entry - v.sl_fixed_pips * PIP
        elif v.sl_mode == "atr":
            sl = entry - v.sl_atr_mult * av if not np.isnan(av) else sl_struct
        elif v.sl_mode == "hybrid_struct_atr":
            atr_sl = entry - v.sl_atr_mult * av if not np.isnan(av) else sl_struct
            sl = max(sl_struct, atr_sl)
        elif v.sl_mode == "structural_capped":
            sl = sl_struct
        else:
            sl = sl_struct

        risk = entry - sl
        if risk <= 0:
            sl = entry - 12 * PIP
            risk = entry - sl

        sl, risk = _cap_risk(entry, sl, "long", v.max_loss_cap_pips)

        tp: float | None
        if v.tp_mode == "fixed_rr":
            tp = entry + v.tp_rr * risk
        elif v.tp_mode == "opp_liquidity":
            tp = ah + buf if ah > entry else entry + v.tp_rr * risk
        elif v.tp_mode == "trail_structure":
            tp = entry + 50.0 * risk
        elif v.tp_mode == "partial_1r_runner":
            tp = entry + v.runner_tp_rr * risk
        else:
            tp = entry + v.tp_rr * risk

        return sl, tp, risk

    sl_struct = max(ah, sweep_high) + buf
    if sl_struct <= entry:
        sl_struct = entry + buf * 2

    if v.sl_mode == "structural":
        sl = sl_struct
    elif v.sl_mode == "fixed_pip":
        sl = entry + v.sl_fixed_pips * PIP
    elif v.sl_mode == "atr":
        sl = entry + v.sl_atr_mult * av if not np.isnan(av) else sl_struct
    elif v.sl_mode == "hybrid_struct_atr":
        atr_sl = entry + v.sl_atr_mult * av if not np.isnan(av) else sl_struct
        sl = min(sl_struct, atr_sl)
    elif v.sl_mode == "structural_capped":
        sl = sl_struct
    else:
        sl = sl_struct

    risk = sl - entry
    if risk <= 0:
        sl = entry + 12 * PIP
        risk = sl - entry

    sl, risk = _cap_risk(entry, sl, "short", v.max_loss_cap_pips)

    if v.tp_mode == "fixed_rr":
        tp = entry - v.tp_rr * risk
    elif v.tp_mode == "opp_liquidity":
        tp = al - buf if al < entry else entry - v.tp_rr * risk
    elif v.tp_mode == "trail_structure":
        tp = entry - 50.0 * risk
    elif v.tp_mode == "partial_1r_runner":
        tp = entry - v.runner_tp_rr * risk
    else:
        tp = entry - v.tp_rr * risk

    return sl, tp, risk


def _adj_entry(raw: float, direction: str, spread_pips: float) -> float:
    if direction == "long":
        return raw + (spread_pips / 2.0) * PIP
    return raw - (spread_pips / 2.0) * PIP


def simulate_variant(ohlcv: pd.DataFrame, setup: TradeSetup, v: ExecutionVariant, atr: pd.Series) -> Phase4SimResult:
    start = entry_bar_index(ohlcv, setup.entry_ts)
    entry = _adj_entry(float(setup.entry), setup.direction, v.spread_pips)

    sl, tp_price, risk0 = compute_sl_tp_risk(setup, ohlcv, atr, entry, start, v)
    if risk0 <= 1e-12:
        return Phase4SimResult(0.0, "flat", ohlcv.index[start], risk0, 0.0)

    sl_live = sl
    tp_live = tp_price
    pos = 1.0
    total_r = 0.0
    be_armed = False
    partial_done = False
    one_r_px = entry + risk0 if setup.direction == "long" else entry - risk0
    mfe_r = 0.0

    exit_ts: pd.Timestamp | None = None
    outcome = "timeout"

    for i in range(start, min(start + v.max_bars, len(ohlcv))):
        ts = ohlcv.index[i]
        bar = ohlcv.iloc[i]
        hi, lo, cl = float(bar["high"]), float(bar["low"]), float(bar["close"])

        if setup.direction == "long":
            mfe_r = max(mfe_r, (hi - entry) / risk0)
        else:
            mfe_r = max(mfe_r, (entry - lo) / risk0)

        # Breakeven
        if v.be_after_r is not None and not be_armed:
            if setup.direction == "long" and hi >= entry + v.be_after_r * risk0:
                sl_live = max(sl_live, entry - 0.5 * PIP)
                be_armed = True
            elif setup.direction == "short" and lo <= entry - v.be_after_r * risk0:
                sl_live = min(sl_live, entry + 0.5 * PIP)
                be_armed = True

        # Partial at 1R
        if (
            v.tp_mode == "partial_1r_runner"
            and v.partial_pct_at_1r
            and not partial_done
            and pos > 0.99
        ):
            hit_1r = hi >= one_r_px if setup.direction == "long" else lo <= one_r_px
            if hit_1r:
                frac = float(v.partial_pct_at_1r)
                move_r = 1.0
                total_r += frac * move_r * pos
                pos *= 1.0 - frac
                partial_done = True
                sl_live = entry - 0.5 * PIP if setup.direction == "long" else entry + 0.5 * PIP
                if setup.direction == "long":
                    tp_live = entry + v.runner_tp_rr * risk0
                else:
                    tp_live = entry - v.runner_tp_rr * risk0

        # Trailing: ratchet stop only on favorable side of entry until BE is armed
        if v.tp_mode == "trail_structure" and i > start:
            av = atr_value(atr, ts)
            if not np.isnan(av):
                eps = 0.5 * PIP
                if setup.direction == "long":
                    trail = hi - v.trail_atr_mult * av
                    if not be_armed:
                        trail = min(trail, entry - eps)
                    sl_live = max(sl_live, trail)
                else:
                    trail = lo + v.trail_atr_mult * av
                    if not be_armed:
                        trail = max(trail, entry + eps)
                    sl_live = min(sl_live, trail)

        # Session close exit (London proxy UTC hour)
        if v.tp_mode == "session_close_ny" and ts.hour >= v.london_exit_utc_hour:
            r_rem = ((cl - entry) / risk0) * pos if setup.direction == "long" else ((entry - cl) / risk0) * pos
            total_r += r_rem
            exit_ts = ts
            outcome = "session_exit"
            break

        # Kill before NY close same UTC day as entry
        if v.kill_ny_utc_hour is not None:
            if ts.date() == ohlcv.index[start].date() and ts.hour >= v.kill_ny_utc_hour:
                r_rem = ((cl - entry) / risk0) * pos if setup.direction == "long" else ((entry - cl) / risk0) * pos
                total_r += r_rem
                exit_ts = ts
                outcome = "ny_kill"
                break

        # Time stop — no expansion
        if v.time_stop_hours is not None:
            hrs = (ts - ohlcv.index[start]).total_seconds() / 3600.0
            if hrs >= v.time_stop_hours and mfe_r < v.min_expansion_r:
                r_rem = ((cl - entry) / risk0) * pos if setup.direction == "long" else ((entry - cl) / risk0) * pos
                total_r += r_rem
                exit_ts = ts
                outcome = "time_stop"
                break

        # Intrabar SL / TP (pessimistic: SL first)
        if setup.direction == "long":
            hit_sl = lo <= sl_live
            tp_active = tp_live if pos > 1e-6 else None
            hit_tp = tp_active is not None and hi >= tp_active
            if hit_sl and hit_tp:
                total_r += -1.0 * pos
                exit_ts = ts
                outcome = "loss"
                break
            if hit_sl:
                total_r += -1.0 * pos
                exit_ts = ts
                outcome = "loss"
                break
            if hit_tp:
                rr_hit = (tp_live - entry) / risk0
                total_r += rr_hit * pos
                exit_ts = ts
                outcome = "win"
                break
        else:
            hit_sl = hi >= sl_live
            hit_tp = tp_live is not None and lo <= tp_live and pos > 1e-6
            if hit_sl and hit_tp:
                total_r += -1.0 * pos
                exit_ts = ts
                outcome = "loss"
                break
            if hit_sl:
                total_r += -1.0 * pos
                exit_ts = ts
                outcome = "loss"
                break
            if hit_tp:
                rr_hit = (entry - tp_live) / risk0
                total_r += rr_hit * pos
                exit_ts = ts
                outcome = "win"
                break

    if outcome == "timeout":
        last_i = min(start + v.max_bars - 1, len(ohlcv) - 1)
        cl = float(ohlcv.iloc[last_i]["close"])
        ts = ohlcv.index[last_i]
        total_r += ((cl - entry) / risk0) * pos if setup.direction == "long" else ((entry - cl) / risk0) * pos
        exit_ts = ts

    if v.slippage_pips > 0 and risk0 > 1e-12:
        rt_slip = 2.0 * v.slippage_pips * PIP
        total_r -= rt_slip / risk0

    return Phase4SimResult(float(total_r), outcome, exit_ts, risk0, float(mfe_r))


def build_execution_variant_grid() -> list[ExecutionVariant]:
    """Curated grid (not full factorial); expands key institutional combinations."""
    out: list[ExecutionVariant] = []

    def add(**kw) -> None:
        name = kw.pop("name")
        out.append(ExecutionVariant(name=name, **kw))

    # Baseline: matches legacy _simulate_one (no spread adj, 200 bars, no cap)
    add(
        name="P3_NATIVE_spread0_capoff_200b",
        sl_mode="structural",
        tp_mode="fixed_rr",
        tp_rr=2.0,
        spread_pips=0.0,
        max_loss_cap_pips=500.0,
        max_bars=200,
        kill_ny_utc_hour=None,
        time_stop_hours=None,
        be_after_r=None,
    )
    # Structural + realistic spread/cap (institutional friction model)
    add(
        name="P3_PARITY_struct_2R",
        sl_mode="structural",
        tp_mode="fixed_rr",
        tp_rr=2.0,
        spread_pips=0.35,
        max_loss_cap_pips=150.0,
        max_bars=200,
        kill_ny_utc_hour=None,
        time_stop_hours=None,
        be_after_r=None,
    )

    sl_modes = ["structural", "fixed_pip", "atr", "hybrid_struct_atr", "structural_capped"]
    tp_rrs = [1.0, 1.5, 2.0, 3.0]
    bes = [None, 0.5, 1.0]
    caps = [35.0, 50.0]

    for sm in sl_modes:
        for tr in tp_rrs:
            for be in bes:
                for cap in caps:
                    if sm == "fixed_pip" and tr != 2.0:
                        continue
                    if sm == "structural_capped" and cap != 35.0:
                        continue
                    nm = f"auto_{sm[:4]}_tp{tr}_be{be}_cap{int(cap)}"
                    add(
                        name=nm,
                        sl_mode=sm,  # type: ignore[arg-type]
                        tp_mode="fixed_rr",
                        tp_rr=tr,
                        be_after_r=be,
                        max_loss_cap_pips=cap,
                        sl_fixed_pips=18.0,
                        sl_atr_mult=1.35,
                        spread_pips=0.35,
                        max_bars=200,
                        kill_ny_utc_hour=None,
                    )

    # Opposing liquidity TP
    add(
        name="opp_liq_struct",
        sl_mode="structural",
        tp_mode="opp_liquidity",
        tp_rr=2.0,
        spread_pips=0.35,
        max_bars=200,
    )
    add(
        name="opp_liq_atrSL",
        sl_mode="atr",
        tp_mode="opp_liquidity",
        tp_rr=2.0,
        sl_atr_mult=1.2,
        spread_pips=0.35,
        max_bars=200,
    )

    # Session / NY / time
    add(
        name="sess_close_L16",
        sl_mode="structural",
        tp_mode="session_close_ny",
        tp_rr=2.0,
        spread_pips=0.35,
        london_exit_utc_hour=16,
        max_bars=200,
    )
    add(
        name="ny_kill_21h",
        sl_mode="structural",
        tp_mode="fixed_rr",
        tp_rr=2.0,
        kill_ny_utc_hour=21,
        spread_pips=0.35,
        max_bars=200,
    )
    add(
        name="time_stop_18h_dead",
        sl_mode="structural",
        tp_mode="fixed_rr",
        tp_rr=2.0,
        time_stop_hours=18.0,
        min_expansion_r=0.2,
        spread_pips=0.35,
        max_bars=200,
    )

    # Trail
    add(
        name="trail_struct",
        sl_mode="structural",
        tp_mode="trail_structure",
        tp_rr=2.5,
        spread_pips=0.35,
        max_bars=200,
    )

    # Partials
    for pct in [0.4, 0.5]:
        add(
            name=f"partial_{int(pct*100)}_runner25",
            sl_mode="structural",
            tp_mode="partial_1r_runner",
            tp_rr=2.0,
            partial_pct_at_1r=pct,
            runner_tp_rr=2.5,
            be_after_r=1.0,
            spread_pips=0.35,
            max_bars=200,
        )

    # Tighter spread stress
    add(
        name="struct_2R_spread08",
        sl_mode="structural",
        tp_mode="fixed_rr",
        tp_rr=2.0,
        spread_pips=0.8,
        max_bars=200,
    )

    # Loser mitigation (section 4): earlier BE, timeouts, tighter cap
    for be_early in (0.3, 0.4):
        add(
            name=f"loser_BE{be_early}_struct_2R",
            sl_mode="structural",
            tp_mode="fixed_rr",
            tp_rr=2.0,
            be_after_r=be_early,
            spread_pips=0.35,
            max_bars=200,
        )
    for hrs in (8.0, 12.0):
        add(
            name=f"loser_timeout{int(hrs)}h_struct_2R",
            sl_mode="structural",
            tp_mode="fixed_rr",
            tp_rr=2.0,
            time_stop_hours=hrs,
            min_expansion_r=0.15,
            spread_pips=0.35,
            max_bars=200,
        )
    add(
        name="loser_cap25_struct_2R",
        sl_mode="structural",
        tp_mode="fixed_rr",
        tp_rr=2.0,
        max_loss_cap_pips=25.0,
        spread_pips=0.35,
        max_bars=200,
    )
    add(
        name="loser_BE03_timeout12h",
        sl_mode="structural",
        tp_mode="fixed_rr",
        tp_rr=2.0,
        be_after_r=0.3,
        time_stop_hours=12.0,
        min_expansion_r=0.12,
        spread_pips=0.35,
        max_bars=200,
    )

    # Winner expansion (section 5): trail / partial already above; extra trail tightness
    add(
        name="trail_struct_tight05",
        sl_mode="structural",
        tp_mode="trail_structure",
        tp_rr=2.5,
        trail_atr_mult=0.5,
        spread_pips=0.35,
        max_bars=200,
    )
    add(
        name="partial50_runner3R_BE05",
        sl_mode="structural",
        tp_mode="partial_1r_runner",
        tp_rr=2.0,
        partial_pct_at_1r=0.5,
        runner_tp_rr=3.0,
        be_after_r=0.5,
        spread_pips=0.35,
        max_bars=200,
    )

    seen: set[str] = set()
    uniq: list[ExecutionVariant] = []
    for v in out:
        if v.name in seen:
            continue
        seen.add(v.name)
        uniq.append(v)
    return uniq


def aggregate_results(rows: list[Phase4SimResult]) -> dict[str, float]:
    if not rows:
        return {"n": 0, "exp": 0.0, "wr": 0.0, "avg_r": 0.0}
    rrs = np.array([r.realized_r for r in rows])
    wins = int((rrs > 0.05).sum())
    n = len(rows)
    return {
        "n": float(n),
        "exp": float(rrs.mean()),
        "wr": wins / n if n else 0.0,
        "avg_r": float(rrs.mean()),
    }


def phase4_eurusd_institutional_baseline() -> ExecutionVariant:
    """Phase 4 winning institutional profile (structural SL, 25 pip max width, 2R TP, spread model)."""
    return ExecutionVariant(
        name="loser_cap25_struct_2R",
        sl_mode="structural",
        tp_mode="fixed_rr",
        tp_rr=2.0,
        max_loss_cap_pips=25.0,
        spread_pips=0.35,
        slippage_pips=0.0,
        max_bars=200,
        kill_ny_utc_hour=None,
        time_stop_hours=None,
        be_after_r=None,
    )

"""
Phase 3 EURUSD institutional filters (hard-wired from Phase 2 winner).

Mirrors ``reports/eurusd_phase2_best_filter.json`` — applied only when the detector
is run with ``symbol=\"EURUSD\"`` and institutional mode enabled.
"""

from __future__ import annotations

from datetime import time
from typing import Any, Literal

import numpy as np
import pandas as pd

from strategy.asian_liquidity_mss import (
    LiquidityMSSConfig,
    fractal_pivot_high,
    fractal_pivot_low,
    _is_bearish_displacement,
    _is_bullish_displacement,
    _segment_range_ref,
)
from strategy.mss_audit_report import _london_ny_flags

# --- Hard-wired winning configuration (eurusd_phase2_best_filter.json) ---
PHASE3_EURUSD_INSTITUTIONAL: dict[str, Any] = {
    "sweep_session_mode": "london_only",
    "min_sweep_penetration_pips": 0.0,
    "min_disp_body_ratio": 0.72,
    "min_disp_range_over_atr": 0.0,
    "min_hours_sweep_to_confirm": 0.0,
    "max_hours_sweep_to_confirm": 48.0,
    "min_risk_pips": 8.0,
    "max_risk_pips": 150.0,
}

PIP_EURUSD = 0.0001
BUF = 3.0 * PIP_EURUSD


def _session_bucket(ts: pd.Timestamp) -> Literal["London", "NY", "overlap", "other"]:
    l_ok, n_ok = _london_ny_flags(ts)
    if l_ok and n_ok:
        return "overlap"
    if l_ok:
        return "London"
    if n_ok:
        return "NY"
    return "other"


def _atr14(ohlcv: pd.DataFrame) -> pd.Series:
    h, l, c = ohlcv["high"], ohlcv["low"], ohlcv["close"]
    pc = c.shift(1)
    tr = pd.concat([(h - l).abs(), (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.rolling(14, min_periods=5).mean()


def _seg_after(base: pd.DataFrame, sweep_ts: pd.Timestamp, cfg: LiquidityMSSConfig) -> pd.DataFrame:
    cols = ["open", "high", "low", "close"]
    end_scan = sweep_ts + pd.Timedelta(hours=cfg.mss_forward_hours)
    return base.loc[(base.index > sweep_ts) & (base.index <= end_scan), cols].copy()


def _bullish_confirm_ts(
    seg_after: pd.DataFrame,
    cfg: LiquidityMSSConfig,
    sweep_ts: pd.Timestamp,
) -> pd.Timestamp | None:
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
    fh = fractal_pivot_high(tail["low"], tail["high"], cfg.fractal_left, cfg.fractal_right)
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
    fl = fractal_pivot_low(tail["low"], tail["high"], cfg.fractal_left, cfg.fractal_right)
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


def _first_disp_body_ratio(
    seg_after: pd.DataFrame,
    cfg: LiquidityMSSConfig,
    sweep_ts: pd.Timestamp,
    bullish: bool,
) -> float | None:
    ref = _segment_range_ref(seg_after)
    for ts, row in seg_after.iterrows():
        if ts <= sweep_ts:
            continue
        ok = (
            _is_bullish_displacement(row, ref, cfg.displacement_body_ratio, cfg.displacement_min_range_ratio)
            if bullish
            else _is_bearish_displacement(row, ref, cfg.displacement_body_ratio, cfg.displacement_min_range_ratio)
        )
        if not ok:
            continue
        o, h, l, cl = float(row["open"]), float(row["high"]), float(row["low"]), float(row["close"])
        rng = h - l
        if rng <= 0:
            return None
        return abs(cl - o) / rng
    return None


def passes_eurusd_institutional_filter(
    ohlcv: pd.DataFrame,
    cfg: LiquidityMSSConfig,
    base: pd.DataFrame,
    ah: float,
    al: float,
    sweep_side: str,
    sweep_ts: pd.Timestamp,
    mss_dir: str,
    seg_after: pd.DataFrame,
    atr: pd.Series,
) -> bool:
    """
    Return True if this candidate setup satisfies Phase 3 institutional rules for EURUSD.
    """
    f = PHASE3_EURUSD_INSTITUTIONAL
    mode = str(f["sweep_session_mode"])

    sess = _session_bucket(sweep_ts)
    if mode == "london_only":
        if sess not in ("London", "overlap"):
            return False
    elif mode == "ny_only":
        if sess not in ("NY", "overlap"):
            return False
    elif mode == "overlap_only":
        if sess != "overlap":
            return False
    elif mode == "no_overlap":
        if sess != "London":
            return False

    if sweep_side == "below" and mss_dir == "bullish":
        confirm_ts = _bullish_confirm_ts(seg_after, cfg, sweep_ts)
        bullish_disp = True
        sweep_pen = max(0.0, (al - float(base.loc[sweep_ts, "low"])) / PIP_EURUSD)
    elif sweep_side == "above" and mss_dir == "bearish":
        confirm_ts = _bearish_confirm_ts(seg_after, cfg, sweep_ts)
        bullish_disp = False
        sweep_pen = max(0.0, (float(base.loc[sweep_ts, "high"]) - ah) / PIP_EURUSD)
    else:
        return False

    if confirm_ts is None:
        return False

    if sweep_pen < float(f["min_sweep_penetration_pips"]):
        return False

    br = _first_disp_body_ratio(seg_after, cfg, sweep_ts, bullish_disp)
    if br is None or br < float(f["min_disp_body_ratio"]):
        return False

    atr_s = float(atr.reindex([sweep_ts]).ffill().bfill().iloc[0]) if len(atr) else float("nan")
    disp_rng_pip = None
    ref = _segment_range_ref(seg_after)
    for ts, row in seg_after.iterrows():
        if ts <= sweep_ts:
            continue
        ok = (
            _is_bullish_displacement(row, ref, cfg.displacement_body_ratio, cfg.displacement_min_range_ratio)
            if bullish_disp
            else _is_bearish_displacement(row, ref, cfg.displacement_body_ratio, cfg.displacement_min_range_ratio)
        )
        if ok:
            h, l = float(row["high"]), float(row["low"])
            disp_rng_pip = (h - l) / PIP_EURUSD
            break
    rng_atr = (disp_rng_pip * PIP_EURUSD) / atr_s if atr_s and atr_s > 0 and disp_rng_pip is not None else 0.0
    if float(f["min_disp_range_over_atr"]) > 0 and rng_atr < float(f["min_disp_range_over_atr"]):
        return False

    hrs = (confirm_ts - sweep_ts).total_seconds() / 3600.0
    if hrs < float(f["min_hours_sweep_to_confirm"]) or hrs > float(f["max_hours_sweep_to_confirm"]):
        return False

    after_confirm = base.loc[base.index > confirm_ts]
    if after_confirm.empty:
        return False
    entry = float(after_confirm.iloc[0]["open"])
    sweep_low = float(base.loc[sweep_ts, "low"])
    sweep_high = float(base.loc[sweep_ts, "high"])

    if mss_dir == "bullish":
        sl = min(al, sweep_low) - BUF
        if sl >= entry:
            sl = entry - BUF * 2
        risk = entry - sl
    else:
        sl = max(ah, sweep_high) + BUF
        if sl <= entry:
            sl = entry + BUF * 2
        risk = sl - entry

    if risk <= 0:
        return False
    risk_pips = risk / PIP_EURUSD
    if risk_pips < float(f["min_risk_pips"]) or risk_pips > float(f["max_risk_pips"]):
        return False

    return True

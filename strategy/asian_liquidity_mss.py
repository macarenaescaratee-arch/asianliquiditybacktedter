"""
Asian liquidity sweep + Market Structure Shift (MSS) signal detector (Phase 3A).

Uses Asian session highs/lows, London/NY sweep windows (post-Asian, same Tokyo day),
wick/body liquidity breaks, candle displacement, and swing fractals for practical MSS.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time
from typing import Literal
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from data.asian_session import AsianSessionWindow, compute_asian_session_extremes

SweepSide = Literal["none", "above", "below"]
MSSDirection = Literal["none", "bullish", "bearish"]
SetupTag = Literal["no setup", "bullish setup candidate", "bearish setup candidate"]


@dataclass(slots=True)
class LiquidityMSSConfig:
    """Tunable detector thresholds."""

    asian: AsianSessionWindow
    fractal_left: int = 2
    fractal_right: int = 2
    displacement_body_ratio: float = 0.55
    # MSS confirmation must follow a displacement candle whose range exceeds this fraction of segment median range
    displacement_min_range_ratio: float = 0.45
    # MSS scan horizon after the sweep (indices with sparse sessions need more forward bars)
    mss_forward_hours: int = 48


def annotate_tokyo_calendar(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """Add ``_tokyo_date`` and ``_tokyo_hour`` for vectorized session filters."""
    dti = ohlcv.index.tz_convert("Asia/Tokyo")
    out = ohlcv.copy()
    out["_tokyo_date"] = pd.Series(dti.date, index=ohlcv.index)
    out["_tokyo_hour"] = pd.Series(dti.hour, index=ohlcv.index, dtype="int16")
    return out


def in_london_or_ny_killzone(ts: datetime | pd.Timestamp) -> bool:
    """
    London morning or NY cash-session morning overlap (local wall clocks).

    Union is intentionally wide enough to cover institutional ``London'' and ``NY'' sweeps
    on hourly data without splitting DST edge cases (acceptable for Phase 3A).
    """
    t = pd.Timestamp(ts)
    if t.tz is None:
        t = t.tz_localize("UTC")
    lon = t.tz_convert("Europe/London")
    ny = t.tz_convert("America/New_York")
    lt = lon.time()
    nt = ny.time()
    london_ok = time(8, 0) <= lt < time(13, 0)
    ny_ok = time(9, 30) <= nt < time(14, 0)
    return london_ok or ny_ok


def post_asian_tokyo_mask(df: pd.DataFrame, session_day: date) -> pd.Series:
    """Same Tokyo calendar day as ``session_day``, excluding 00:00–09:00 Tokyo (Asian build)."""
    return (df["_tokyo_date"] == session_day) & (df["_tokyo_hour"] >= 9)


def build_london_ny_killzone_mask(index: pd.DatetimeIndex) -> pd.Series:
    """Pre-compute London/NY killzone flags (one pass over all bars)."""
    flags = np.fromiter(
        (in_london_or_ny_killzone(ts) for ts in index),
        dtype=bool,
        count=len(index),
    )
    return pd.Series(flags, index=index, dtype=bool)


def sweep_session_mask(
    df: pd.DataFrame,
    session_day: date,
    killzone: pd.Series,
) -> pd.Series:
    """Post-Asian Tokyo ``session_day`` bars that also fall in London/NY kill zones."""
    return post_asian_tokyo_mask(df, session_day) & killzone


def fractal_pivot_high(low: pd.Series, high: pd.Series, left: int, right: int) -> pd.Series:
    """Boolean Series: strict fractal swing high (center strictly above immediate neighbours)."""
    out = pd.Series(False, index=high.index)
    n = len(high)
    if n < left + right + 1:
        return out
    arr_h = high.to_numpy(dtype=float)
    for i in range(left, n - right):
        window = arr_h[i - left : i + right + 1]
        hi = arr_h[i]
        if hi >= np.max(window) and hi > arr_h[i - 1] and hi > arr_h[i + 1]:
            out.iloc[i] = True
    return out


def fractal_pivot_low(low: pd.Series, high: pd.Series, left: int, right: int) -> pd.Series:
    """Boolean Series: strict fractal swing low."""
    out = pd.Series(False, index=low.index)
    n = len(low)
    if n < left + right + 1:
        return out
    arr_l = low.to_numpy(dtype=float)
    for i in range(left, n - right):
        window = arr_l[i - left : i + right + 1]
        lo = arr_l[i]
        if lo <= np.min(window) and lo < arr_l[i - 1] and lo < arr_l[i + 1]:
            out.iloc[i] = True
    return out


def _segment_range_ref(seg: pd.DataFrame) -> float:
    rng = (seg["high"] - seg["low"]).astype(float)
    med = float(np.nanmedian(rng.to_numpy())) if len(rng) else 0.0
    return med if med > 0 else float(np.nanmean(rng.to_numpy()) or 0.0)


def _is_bearish_displacement(row: pd.Series, ref_range: float, body_ratio: float, min_rr: float) -> bool:
    o, h, l, c = float(row["open"]), float(row["high"]), float(row["low"]), float(row["close"])
    rng = h - l
    if rng <= 0:
        return False
    body = abs(c - o)
    if c >= o:
        return False
    if body / rng < body_ratio:
        return False
    if ref_range > 0 and rng < ref_range * min_rr:
        return False
    return True


def _is_bullish_displacement(row: pd.Series, ref_range: float, body_ratio: float, min_rr: float) -> bool:
    o, h, l, c = float(row["open"]), float(row["high"]), float(row["low"]), float(row["close"])
    rng = h - l
    if rng <= 0:
        return False
    body = abs(c - o)
    if c <= o:
        return False
    if body / rng < body_ratio:
        return False
    if ref_range > 0 and rng < ref_range * min_rr:
        return False
    return True


def _first_liquidity_sweep(
    seg: pd.DataFrame,
    ah: float,
    al: float,
) -> tuple[SweepSide, pd.Timestamp | None]:
    """
    First chronological liquidity sweep in ``seg`` (must be in sweep session already).

    Above: ``high > AH``; below: ``low < AL``. Tie-break: earlier timestamp; if one bar does both, prefer ``above``.
    """
    if seg.empty:
        return "none", None
    above_ix = seg.index[seg["high"].astype(float) > ah]
    below_ix = seg.index[seg["low"].astype(float) < al]
    t_above = above_ix.min() if len(above_ix) else None
    t_below = below_ix.min() if len(below_ix) else None
    if t_above is None and t_below is None:
        return "none", None
    if t_above is not None and t_below is not None:
        if t_above == t_below:
            return "above", t_above
        return ("above", t_above) if t_above < t_below else ("below", t_below)
    if t_above is not None:
        return "above", t_above
    return "below", t_below


def _bearish_mss_after_sweep_up(
    seg_after: pd.DataFrame,
    cfg: LiquidityMSSConfig,
    sweep_idx: pd.Timestamp,
) -> bool:
    """
    After sweep above: displacement down, then break below last fractal swing low (post-displacement).
    """
    if seg_after.empty:
        return False
    ref_range = _segment_range_ref(seg_after)
    # displacement must occur after sweep
    disp_pos = None
    for i, (ts, row) in enumerate(seg_after.iterrows()):
        if ts <= sweep_idx:
            continue
        if _is_bearish_displacement(row, ref_range, cfg.displacement_body_ratio, cfg.displacement_min_range_ratio):
            disp_pos = i
            break
    if disp_pos is None:
        return False
    tail = seg_after.iloc[disp_pos:]
    lows = tail["low"]
    highs = tail["high"]
    fl = fractal_pivot_low(lows, highs, cfg.fractal_left, cfg.fractal_right)
    # require at least one fractal low after displacement, then a close below that low (structure break)
    post = tail[fl]
    if post.empty:
        # fallback: break below running minimum low after displacement
        anchor_low = float(tail["low"].iloc[0])
        for _, row in tail.iloc[1:].iterrows():
            if float(row["close"]) < anchor_low:
                return True
        return False
    last_anchor = float(post["low"].iloc[-1])
    # confirmation: any subsequent close strictly below the last recorded swing low
    after_last = tail.loc[tail.index > post.index[-1]]
    if after_last.empty:
        return False
    return bool((after_last["close"].astype(float) < last_anchor).any())


def _bullish_mss_after_sweep_down(
    seg_after: pd.DataFrame,
    cfg: LiquidityMSSConfig,
    sweep_idx: pd.Timestamp,
) -> bool:
    """After sweep below: displacement up, then break above last fractal swing high (post-displacement)."""
    if seg_after.empty:
        return False
    ref_range = _segment_range_ref(seg_after)
    disp_pos = None
    for i, (ts, row) in enumerate(seg_after.iterrows()):
        if ts <= sweep_idx:
            continue
        if _is_bullish_displacement(row, ref_range, cfg.displacement_body_ratio, cfg.displacement_min_range_ratio):
            disp_pos = i
            break
    if disp_pos is None:
        return False
    tail = seg_after.iloc[disp_pos:]
    lows = tail["low"]
    highs = tail["high"]
    fh = fractal_pivot_high(lows, highs, cfg.fractal_left, cfg.fractal_right)
    post = tail[fh]
    if post.empty:
        anchor_high = float(tail["high"].iloc[0])
        for _, row in tail.iloc[1:].iterrows():
            if float(row["close"]) > anchor_high:
                return True
        return False
    last_anchor = float(post["high"].iloc[-1])
    after_last = tail.loc[tail.index > post.index[-1]]
    if after_last.empty:
        return False
    return bool((after_last["close"].astype(float) > last_anchor).any())


def detect_asian_liquidity_mss(
    ohlcv: pd.DataFrame,
    cfg: LiquidityMSSConfig,
    *,
    symbol: str | None = None,
    eurusd_institutional: bool = True,
) -> pd.DataFrame:
    """
    Build a daily summary table keyed by Asian ``session_date`` (Tokyo).

    Parameters
    ----------
    symbol
        When ``\"EURUSD\"`` and ``eurusd_institutional`` is True, applies Phase 3
        institutional filters so only optimized setups receive candidate tags.
    eurusd_institutional
        If False, EURUSD uses the same raw MSS rules as other symbols (for baselines).

    Columns
    -------
    session_date, asian_high, asian_low, sweep_side, mss_direction, tag
    """
    if not isinstance(ohlcv.index, pd.DatetimeIndex) or ohlcv.index.tz is None:
        raise ValueError("ohlcv must have tz-aware DatetimeIndex (UTC).")

    df = annotate_tokyo_calendar(ohlcv)
    killzone = build_london_ny_killzone_mask(df.index)
    extremes = compute_asian_session_extremes(ohlcv, cfg.asian)
    rows: list[dict] = []

    apply_eurusd_filters = bool(
        eurusd_institutional and symbol is not None and str(symbol).strip().upper() == "EURUSD"
    )
    atr_ser = None
    if apply_eurusd_filters:
        from strategy.eurusd_institutional_filters import _atr14

        atr_ser = _atr14(ohlcv)

    cols_ohlc = ["open", "high", "low", "close"]
    base_ohlc = df.loc[:, cols_ohlc]

    for _, xr in extremes.iterrows():
        D = xr["session_date"]
        ah, al = float(xr["asian_high"]), float(xr["asian_low"])
        sm = sweep_session_mask(df, D, killzone)
        seg = df.loc[sm, ["open", "high", "low", "close"]].copy()
        sweep_side, sweep_ts = _first_liquidity_sweep(seg, ah, al)

        mss_dir: MSSDirection = "none"
        tag: SetupTag = "no setup"

        if sweep_side == "none":
            rows.append(
                {
                    "session_date": D,
                    "asian_high": ah,
                    "asian_low": al,
                    "sweep_side": sweep_side,
                    "mss_direction": mss_dir,
                    "tag": tag,
                }
            )
            continue

        cols = ["open", "high", "low", "close"]
        if sweep_ts is None:
            seg_after = df.loc[:, cols].iloc[0:0]
        else:
            end_scan = sweep_ts + pd.Timedelta(hours=cfg.mss_forward_hours)
            seg_after = df.loc[(df.index > sweep_ts) & (df.index <= end_scan), cols].copy()

        if sweep_side == "above":
            ok = _bearish_mss_after_sweep_up(seg_after, cfg, sweep_ts) if sweep_ts is not None else False
            if ok:
                mss_dir = "bearish"
                tag = "bearish setup candidate"
        elif sweep_side == "below":
            ok = _bullish_mss_after_sweep_down(seg_after, cfg, sweep_ts) if sweep_ts is not None else False
            if ok:
                mss_dir = "bullish"
                tag = "bullish setup candidate"

        if apply_eurusd_filters and atr_ser is not None and tag != "no setup":
            from strategy.eurusd_institutional_filters import passes_eurusd_institutional_filter

            if not passes_eurusd_institutional_filter(
                ohlcv,
                cfg,
                base_ohlc,
                ah,
                al,
                sweep_side,
                sweep_ts,
                mss_dir,
                seg_after,
                atr_ser,
            ):
                mss_dir = "none"
                tag = "no setup"

        rows.append(
            {
                "session_date": D,
                "asian_high": ah,
                "asian_low": al,
                "sweep_side": sweep_side,
                "mss_direction": mss_dir,
                "tag": tag,
            }
        )

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values("session_date").reset_index(drop=True)
    return out


def default_liquidity_mss_config() -> LiquidityMSSConfig:
    """Defaults aligned with ``config.DEFAULT_ASIAN_*`` (Tokyo 00:00–09:00)."""
    from datetime import time as dtime

    from config import DEFAULT_ASIAN_END, DEFAULT_ASIAN_START, DEFAULT_ASIAN_TZ

    def _parse_hhmm(s: str) -> dtime:
        h, m = s.split(":")
        return dtime(int(h), int(m))

    aw = AsianSessionWindow(
        timezone=DEFAULT_ASIAN_TZ,
        asian_start=_parse_hhmm(DEFAULT_ASIAN_START),
        asian_end=_parse_hhmm(DEFAULT_ASIAN_END),
    )
    return LiquidityMSSConfig(asian=aw)

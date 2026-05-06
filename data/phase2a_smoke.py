"""
Phase 2A smoke test: synthetic OHLCV + Asian session extremes + sample printout.

Run from project root::

    python -m data.phase2a_smoke
"""

from __future__ import annotations

import tempfile
from datetime import time
from pathlib import Path

import pandas as pd

from data.asian_session import (
    AsianSessionWindow,
    compute_asian_session_extremes,
    print_sample_asian_days,
)
from data.loader import load_ohlcv_csv


def build_synthetic_ohlcv() -> pd.DataFrame:
    """
    Hourly UTC bars with **known** spikes on the first Asian-window bar of each Tokyo day.

    Window in tests: 00:00–09:00 ``Asia/Tokyo`` (same calendar day, ``start < end``).
    """
    idx = pd.date_range("2024-01-01 00:00", periods=24 * 8, freq="1h", tz="UTC")
    base = 100.0
    high = pd.Series(base, index=idx)
    low = pd.Series(base - 0.5, index=idx)

    local = idx.tz_convert("Asia/Tokyo")
    seen: set = set()
    for i, ts_loc in enumerate(local):
        if not (0 <= ts_loc.hour < 9):
            continue
        dkey = ts_loc.date()
        if dkey in seen:
            continue
        seen.add(dkey)
        tag = len(seen)
        bar_ts = idx[i]
        high.loc[bar_ts] = base + tag * 10.0
        low.loc[bar_ts] = base - 0.5 - tag * 2.0

    mid = (high + low) / 2.0
    return pd.DataFrame({"open": mid, "high": high, "low": low, "close": mid}, index=idx)


def verify_csv_loader_roundtrip() -> None:
    """Write a tiny CSV, reload, assert UTC index and column contract."""
    df = build_synthetic_ohlcv().iloc[:48].copy()
    df.index.name = "datetime"
    out = df.reset_index()

    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "mini.csv"
        out.to_csv(
            p,
            index=False,
            columns=["datetime", "open", "high", "low", "close"],
        )
        back = load_ohlcv_csv(p)
        assert back.index.tz is not None
        assert list(back.columns)[:4] == ["open", "high", "low", "close"]


def run_phase2a_sample() -> pd.DataFrame:
    """Build synthetic data, compute Asian extremes, print five sample days."""
    verify_csv_loader_roundtrip()
    ohlcv = build_synthetic_ohlcv()
    window = AsianSessionWindow(
        timezone="Asia/Tokyo",
        asian_start=time(0, 0),
        asian_end=time(9, 0),
    )
    extremes = compute_asian_session_extremes(ohlcv, window)
    print_sample_asian_days(extremes, n=5)
    return extremes


if __name__ == "__main__":
    run_phase2a_sample()

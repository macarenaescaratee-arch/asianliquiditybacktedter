"""
Institutional-style historical OHLC downloader using Dukascopy's public data feed (via
``dukascopy-python``).

Outputs CSV files compatible with ``data.loader.load_ohlcv_csv``:
``datetime, open, high, low, close`` (UTC, timezone-aware ISO strings).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import dukascopy_python as dk
import pandas as pd

from config import DATA_DIR
from data.dukascopy_symbols import dukascopy_instrument_for
from data.symbols import SUPPORTED_OHLC_SYMBOLS, default_raw_csv_path
from utils.helpers import get_logger

LOG = get_logger(__name__)


@dataclass(slots=True)
class DukascopyDownloadConfig:
    """User-facing download window and candle settings."""

    start: datetime
    end: datetime
    interval: Any = dk.INTERVAL_HOUR_1
    offer_side: Any = dk.OFFER_SIDE_BID

    def validated(self) -> DukascopyDownloadConfig:
        """Ensure UTC-aware bounds and ``start < end``."""
        start = _ensure_utc(self.start)
        end = _ensure_utc(self.end)
        if start >= end:
            raise ValueError(f"start must be before end; got start={start}, end={end}")
        return DukascopyDownloadConfig(
            start=start, end=end, interval=self.interval, offer_side=self.offer_side
        )


def _ensure_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def default_three_year_window(
    *,
    end: datetime | None = None,
) -> tuple[datetime, datetime]:
    """``end`` (default: now UTC) and ``start`` = ``end`` − 3 years."""
    e = _ensure_utc(end or datetime.now(timezone.utc))
    s = e - timedelta(days=365 * 3)
    return s, e


def fetch_dukascopy_ohlcv(
    symbol: str,
    cfg: DukascopyDownloadConfig,
) -> pd.DataFrame:
    """
    Download OHLCV from Dukascopy for one symbol.

    Returns a **sorted** DataFrame indexed by UTC timestamps with columns
    ``open, high, low, close, volume`` (volume retained until CSV export strips it).
    """
    c = cfg.validated()
    instrument = dukascopy_instrument_for(symbol)
    LOG.info("Fetching %s from Dukascopy: %s → %s", symbol, c.start, c.end)
    df = dk.fetch(
        instrument,
        c.interval,
        c.offer_side,
        c.start,
        c.end,
    )
    if df is None or df.empty:
        raise RuntimeError(f"No rows returned for {symbol} in the requested window.")
    return normalize_dukascopy_dataframe(df)


def normalize_dukascopy_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enforce UTC DatetimeIndex, monotonic order, and strip duplicate timestamps.

    The upstream package typically names the index ``timestamp`` and uses OHLCV columns.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        if "timestamp" in df.columns:
            df = df.set_index("timestamp")
        else:
            raise ValueError("Expected DatetimeIndex or a timestamp column.")
    out = df.copy()
    if out.index.tz is None:
        out.index = out.index.tz_localize(timezone.utc)
    else:
        out.index = out.index.tz_convert(timezone.utc)
    out.index.name = "timestamp"
    out = out.sort_index()
    out = out[~out.index.duplicated(keep="last")]
    for col in ("open", "high", "low", "close"):
        if col not in out.columns:
            raise ValueError(f"Missing column {col!r}; got {list(out.columns)}")
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def clean_missing_ohlc_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows with incomplete OHLC; log how many were removed."""
    ohlc = ("open", "high", "low", "close")
    before = len(df)
    cleaned = df.dropna(subset=list(ohlc), how="any")
    removed = before - len(cleaned)
    if removed:
        LOG.warning("Dropped %s rows with NaN OHLC (of %s).", removed, before)
    return cleaned


def to_csv_ready_frame(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a frame with a ``datetime`` column (ISO-8601 UTC strings) and OHLC only.

    Matches ``data.loader.load_ohlcv_csv`` expectations.
    """
    out = df.copy()
    out = out.reset_index()
    ts_col = "timestamp" if "timestamp" in out.columns else out.columns[0]
    out = out.rename(columns={ts_col: "datetime"})
    out["datetime"] = pd.to_datetime(out["datetime"], utc=True)
    # Stable, explicit UTC formatting for CSV readability
    out["datetime"] = out["datetime"].dt.strftime("%Y-%m-%d %H:%M:%S%z")
    keep = ["datetime", "open", "high", "low", "close"]
    return out[keep]


def save_raw_csv(df_csv_ready: pd.DataFrame, path: str | Path) -> Path:
    """Write CSV atomically (write temp then replace)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    df_csv_ready.to_csv(tmp, index=False)
    tmp.replace(p)
    LOG.info("Wrote %s rows to %s", len(df_csv_ready), p)
    return p.resolve()


def download_symbol_to_raw(
    symbol: str,
    cfg: DukascopyDownloadConfig,
    *,
    data_dir: Path | None = None,
    output_path: str | Path | None = None,
) -> Path:
    """Fetch, clean, normalize, and save one symbol to ``data/raw/{SYMBOL}.csv``."""
    base = data_dir if data_dir is not None else DATA_DIR
    raw = fetch_dukascopy_ohlcv(symbol, cfg)
    raw = clean_missing_ohlc_rows(raw)
    csv_df = to_csv_ready_frame(raw)
    dest = Path(output_path) if output_path is not None else default_raw_csv_path(symbol, base)
    return save_raw_csv(csv_df, dest)


def download_all_supported(
    cfg: DukascopyDownloadConfig,
    *,
    symbols: frozenset[str] | None = None,
    data_dir: Path | None = None,
) -> dict[str, Path]:
    """Download every supported symbol (or a subset) and return symbol → path mapping."""
    symset = symbols if symbols is not None else SUPPORTED_OHLC_SYMBOLS
    out: dict[str, Path] = {}
    for sym in sorted(symset):
        out[sym] = download_symbol_to_raw(sym, cfg, data_dir=data_dir)
    return out


def _configure_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(level=level, format="%(levelname)s %(name)s: %(message)s")


def main(argv: list[str] | None = None) -> int:
    """CLI: ``python -m data.downloader`` with optional ISO start/end (UTC)."""
    import argparse

    _configure_logging()
    parser = argparse.ArgumentParser(description="Download Dukascopy OHLC into data/raw/")
    parser.add_argument(
        "--start",
        type=str,
        default=None,
        help="Start datetime ISO-8601 (UTC). Default: end − 3 years.",
    )
    parser.add_argument(
        "--end",
        type=str,
        default=None,
        help="End datetime ISO-8601 (UTC). Default: now.",
    )
    parser.add_argument(
        "--symbols",
        type=str,
        default=",".join(sorted(SUPPORTED_OHLC_SYMBOLS)),
        help="Comma-separated symbols (default: all supported).",
    )
    args = parser.parse_args(argv)

    if args.end:
        end = datetime.fromisoformat(args.end.replace("Z", "+00:00"))
    else:
        end = datetime.now(timezone.utc)

    if args.start:
        start = datetime.fromisoformat(args.start.replace("Z", "+00:00"))
    else:
        start = end - timedelta(days=365 * 3)

    cfg = DukascopyDownloadConfig(start=start, end=end).validated()
    want = frozenset(s.strip().upper() for s in args.symbols.split(",") if s.strip())
    for s in want:
        if s not in SUPPORTED_OHLC_SYMBOLS:
            raise SystemExit(f"Unknown symbol: {s}")

    paths = download_all_supported(cfg, symbols=want)
    for sym, path in paths.items():
        print(f"{sym} -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

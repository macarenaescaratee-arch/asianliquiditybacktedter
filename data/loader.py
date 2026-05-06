"""
Historical OHLC(V) CSV loading and normalization.

Phase 2A: CSV with columns ``datetime, open, high, low, close`` (optional ``volume``).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from data.symbols import default_raw_csv_path, validate_symbol

REQUIRED_OHLC_COLUMNS: frozenset[str] = frozenset(
    {"datetime", "open", "high", "low", "close"}
)


def load_ohlcv_csv(
    path: str | Path,
    *,
    datetime_column: str = "datetime",
    sort: bool = True,
    drop_duplicate_times: bool = True,
) -> pd.DataFrame:
    """
    Load OHLC from CSV, normalize datetimes to UTC, sort ascending.

    Parameters
    ----------
    path
        Filesystem path to CSV.
    datetime_column
        Name of the timestamp column **after** lowercasing headers (default ``datetime``).
    sort
        If True, sort by index ascending.
    drop_duplicate_times
        If True, keep last row per duplicate timestamp.

    Returns
    -------
    DataFrame indexed by UTC DatetimeIndex with ``open, high, low, close`` (+ ``volume`` if present).
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"CSV not found: {p}")

    df = pd.read_csv(p)
    df.columns = [str(c).strip().lower() for c in df.columns]

    if datetime_column.lower() not in df.columns:
        raise ValueError(
            f"Missing datetime column {datetime_column!r}. Found: {list(df.columns)}."
        )
    if not REQUIRED_OHLC_COLUMNS <= set(df.columns):
        missing = sorted(REQUIRED_OHLC_COLUMNS - set(df.columns))
        raise ValueError(f"Missing required columns: {missing}. Found: {list(df.columns)}")

    dt_key = datetime_column.lower()
    dt = pd.to_datetime(df[dt_key], utc=True, format="mixed")
    df = df.drop(columns=[dt_key])
    df.insert(0, "_dt", dt)
    df = df.set_index("_dt").sort_index()
    df.index.name = "datetime"

    for col in ("open", "high", "low", "close"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if "volume" in df.columns:
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce")

    ohlc_cols = ("open", "high", "low", "close")
    if df[list(ohlc_cols)].isna().any().any():
        bad = df[df[list(ohlc_cols)].isna().any(axis=1)]
        raise ValueError(f"Non-numeric or missing OHLC rows after parse, e.g.:\n{bad.head()}")

    extra = [c for c in df.columns if c not in ohlc_cols and c != "volume"]
    if extra:
        df = df.drop(columns=extra, errors="ignore")

    if sort:
        df = df.sort_index()

    if drop_duplicate_times:
        df = df[~df.index.duplicated(keep="last")]

    return df


def load_symbol_ohlcv_csv(
    symbol: str,
    path: str | Path | None = None,
    *,
    data_dir: Path | None = None,
) -> pd.DataFrame:
    """
    Load OHLCV for a supported symbol (EURUSD, GBPUSD, XAUUSD, NAS100).

    If ``path`` is omitted, reads ``{DATA_DIR}/raw/{SYMBOL}.csv`` from config.
    """
    from config import DATA_DIR

    base = data_dir if data_dir is not None else DATA_DIR
    validate_symbol(symbol)
    csv_path = Path(path) if path is not None else default_raw_csv_path(symbol, base)
    return load_ohlcv_csv(csv_path)


def resolve_data_path(relative: str | Path) -> Path:
    """Resolve a path relative to the project ``data/`` directory."""
    from config import DATA_DIR

    return (DATA_DIR / Path(relative)).resolve()

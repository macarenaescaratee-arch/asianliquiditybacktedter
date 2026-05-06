"""
Supported dataset symbols for Phase 2A loaders.

Maps logical symbols to default on-disk layouts under ``data/raw/``.
"""

from __future__ import annotations

from pathlib import Path

SUPPORTED_OHLC_SYMBOLS: frozenset[str] = frozenset(
    {"EURUSD", "GBPUSD", "XAUUSD", "NAS100"}
)


def validate_symbol(symbol: str) -> str:
    """Return uppercased symbol or raise ValueError if unsupported."""
    s = symbol.strip().upper()
    if s not in SUPPORTED_OHLC_SYMBOLS:
        raise ValueError(
            f"Unsupported symbol {symbol!r}. Expected one of {sorted(SUPPORTED_OHLC_SYMBOLS)}."
        )
    return s


def default_raw_csv_path(symbol: str, data_dir: Path) -> Path:
    """Default CSV path: ``{data_dir}/raw/{SYMBOL}.csv``."""
    return (data_dir / "raw" / f"{validate_symbol(symbol)}.csv").resolve()

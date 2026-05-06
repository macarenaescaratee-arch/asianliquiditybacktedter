"""
Data package: ingestion, cleaning, storage, and cataloging of market data.

Phase 2A: ``loader`` (CSV OHLCV), ``symbols`` (EURUSD/GBPUSD/XAUUSD/NAS100),
``asian_session`` (session high/low table), and ``phase2a_smoke`` for a quick check.
"""

from data.loader import load_ohlcv_csv, load_symbol_ohlcv_csv
from data.asian_session import AsianSessionWindow, compute_asian_session_extremes

__all__ = [
    "load_ohlcv_csv",
    "load_symbol_ohlcv_csv",
    "AsianSessionWindow",
    "compute_asian_session_extremes",
]

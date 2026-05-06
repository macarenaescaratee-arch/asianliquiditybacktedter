"""
Global configuration for AsianLiquidityBacktester.

Future use: centralize paths, API keys (via environment variables), default backtest
parameters, session/session timezone defaults for Asian session liquidity studies,
and feature flags. Import this module instead of scattering constants across packages.
"""

from __future__ import annotations

import os
from pathlib import Path

# Project root (directory containing this file)
PROJECT_ROOT: Path = Path(__file__).resolve().parent

# --- Data layout (future: raw vs processed, vendor feeds) ---
DATA_DIR: Path = PROJECT_ROOT / "data"
REPORTS_DIR: Path = PROJECT_ROOT / "reports"

# --- Runtime ---
# Default timezone for session-based strategies (e.g. Tokyo/Hong Kong liquidity windows)
DEFAULT_TIMEZONE: str = os.environ.get("ALB_TIMEZONE", "UTC")

# --- Asian session defaults (local wall-clock in DEFAULT_ASIAN_TZ) ---
DEFAULT_ASIAN_TZ: str = os.environ.get("ALB_ASIAN_TZ", "Asia/Tokyo")
# HH:MM in 24h; override via env if needed for research presets
DEFAULT_ASIAN_START: str = os.environ.get("ALB_ASIAN_START", "00:00")
DEFAULT_ASIAN_END: str = os.environ.get("ALB_ASIAN_END", "09:00")

# --- Backtest defaults (placeholders for Phase 2+) ---
DEFAULT_INITIAL_CASH: float = float(os.environ.get("ALB_INITIAL_CASH", "100_000"))
DEFAULT_COMMISSION_BPS: float = float(os.environ.get("ALB_COMMISSION_BPS", "5"))

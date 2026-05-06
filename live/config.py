"""Live bot defaults (paths, warmup)."""

from __future__ import annotations

from pathlib import Path

from config import PROJECT_ROOT

LOG_DIR: Path = PROJECT_ROOT / "logs"
DEFAULT_LIVE_LOG_PATH: Path = LOG_DIR / "eurusd_institutional_live.jsonl"
DEFAULT_DAEMON_STATE_PATH: Path = LOG_DIR / "eurusd_daemon_state.json"
DEFAULT_WATCHDOG_PATH: Path = LOG_DIR / "eurusd_daemon_watchdog.json"
DEFAULT_DAEMON_LOCK_PATH: Path = LOG_DIR / "eurusd_daemon.lock"
DEFAULT_METRICS_PATH: Path = LOG_DIR / "eurusd_daemon_metrics.json"
DEFAULT_METRICS_HISTORY_PATH: Path = LOG_DIR / "eurusd_daemon_metrics.jsonl"
MIN_OHLC_BARS_WARMUP: int = 900

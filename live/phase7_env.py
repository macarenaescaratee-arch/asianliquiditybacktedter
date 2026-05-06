"""Environment configuration for Phase 7 live/demo deployment."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class Phase7OandaConfig:
    api_token: str
    account_id: str
    practice: bool
    instrument: str = "EUR_USD"
    poll_seconds: float = 15.0
    risk_fraction: float = 0.005
    default_units: float = 10000.0
    candle_bootstrap: int = 1200
    live_min_bars: int = 900
    poll_count: int = 120
    max_retry_backoff_seconds: float = 120.0
    alert_consecutive_failures: int = 5

    @classmethod
    def from_environ(cls) -> Phase7OandaConfig:
        token = os.environ.get("OANDA_API_TOKEN", "").strip()
        account = os.environ.get("OANDA_ACCOUNT_ID", "").strip()
        env = os.environ.get("OANDA_ENV", "practice").strip().lower()
        practice = env != "live"
        inst = os.environ.get("OANDA_INSTRUMENT", "EUR_USD").strip()
        poll = float(os.environ.get("OANDA_POLL_SECONDS", "15"))
        risk = float(os.environ.get("OANDA_RISK_FRACTION", "0.005"))
        units = float(os.environ.get("OANDA_DEFAULT_UNITS", "10000"))
        boot = int(os.environ.get("OANDA_BOOTSTRAP_CANDLES", "1200"))
        min_bars = int(os.environ.get("LIVE_MIN_BARS", "900"))
        poll_count = int(os.environ.get("OANDA_POLL_COUNT", "120"))
        max_backoff = float(os.environ.get("DAEMON_MAX_BACKOFF_SECONDS", "120"))
        alert_fails = int(os.environ.get("DAEMON_ALERT_FAILURES", "5"))
        return cls(
            api_token=token,
            account_id=account,
            practice=practice,
            instrument=inst,
            poll_seconds=poll,
            risk_fraction=risk,
            default_units=units,
            candle_bootstrap=boot,
            live_min_bars=min_bars,
            poll_count=poll_count,
            max_retry_backoff_seconds=max_backoff,
            alert_consecutive_failures=alert_fails,
        )

    def validate(self) -> None:
        if not self.api_token or not self.account_id:
            raise RuntimeError(
                "Set OANDA_API_TOKEN and OANDA_ACCOUNT_ID for live execution "
                "(practice: OANDA_ENV=practice, default)."
            )
        if self.candle_bootstrap < self.live_min_bars:
            raise RuntimeError(
                f"OANDA_BOOTSTRAP_CANDLES ({self.candle_bootstrap}) must be >= "
                f"LIVE_MIN_BARS ({self.live_min_bars})."
            )
        if self.poll_count < 30:
            raise RuntimeError("OANDA_POLL_COUNT must be >= 30 for gap recovery resilience.")

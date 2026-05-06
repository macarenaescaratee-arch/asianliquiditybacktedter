"""
Run the institutional EURUSD bot against OANDA (practice or live).

Examples::

    export OANDA_API_TOKEN=...
    export OANDA_ACCOUNT_ID=...
    python -m live.run_daemon --paper
    python -m live.run_daemon
"""

from __future__ import annotations

import argparse
import logging
import signal
from pathlib import Path

from live.config import (
    DEFAULT_DAEMON_LOCK_PATH,
    DEFAULT_DAEMON_STATE_PATH,
    DEFAULT_LIVE_LOG_PATH,
    DEFAULT_METRICS_HISTORY_PATH,
    DEFAULT_METRICS_PATH,
    DEFAULT_WATCHDOG_PATH,
)
from live.daemon import run_forever


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    def _handle_term(_sig, _frame) -> None:
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, _handle_term)
    p = argparse.ArgumentParser(description="OANDA-backed EURUSD institutional daemon")
    p.add_argument(
        "--paper",
        action="store_true",
        help="Poll OANDA for candles but route orders to the paper broker (no venue orders).",
    )
    p.add_argument("--log", type=Path, default=DEFAULT_LIVE_LOG_PATH, help="JSONL log path")
    p.add_argument(
        "--state",
        type=Path,
        default=DEFAULT_DAEMON_STATE_PATH,
        help="Durable daemon runtime state JSON path.",
    )
    p.add_argument(
        "--watchdog",
        type=Path,
        default=DEFAULT_WATCHDOG_PATH,
        help="Watchdog heartbeat status JSON path.",
    )
    p.add_argument(
        "--max-polls",
        type=int,
        default=None,
        metavar="N",
        help="Exit cleanly after N polling iterations (default: run until interrupted).",
    )
    p.add_argument("--lock", type=Path, default=DEFAULT_DAEMON_LOCK_PATH, help="Singleton PID lock file path.")
    p.add_argument("--metrics", type=Path, default=DEFAULT_METRICS_PATH, help="Metrics snapshot JSON path.")
    p.add_argument(
        "--metrics-history",
        type=Path,
        default=DEFAULT_METRICS_HISTORY_PATH,
        help="Rolling metrics history JSONL path.",
    )
    args = p.parse_args()
    try:
        run_forever(
            paper=args.paper,
            log_path=args.log,
            max_polls=args.max_polls,
            state_path=args.state,
            watchdog_path=args.watchdog,
            lock_path=args.lock,
            metrics_path=args.metrics,
            metrics_history_path=args.metrics_history,
        )
    except KeyboardInterrupt:
        logging.getLogger("live.run_daemon").info("Stopped by user")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

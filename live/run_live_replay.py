"""
CLI: replay historical EURUSD hourly CSV through the live bot skeleton.

Example::

    python -m live.run_live_replay
    python -m live.run_live_replay --csv /path/to/EURUSD.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

from config import DATA_DIR
from data.symbols import default_raw_csv_path
from live.config import DEFAULT_LIVE_LOG_PATH, MIN_OHLC_BARS_WARMUP
from live.runner import LiveEURUSDBot
from live.ingestion import CsvReplaySource


def main() -> int:
    p = argparse.ArgumentParser(description="Replay EURUSD through live institutional bot")
    p.add_argument(
        "--csv",
        type=Path,
        default=default_raw_csv_path("EURUSD", DATA_DIR),
        help="OHLCV CSV path",
    )
    p.add_argument(
        "--tail",
        type=int,
        default=500,
        help="Only replay the last N rows (0 = entire CSV; large N is slow: detect runs on full buffer each bar).",
    )
    p.add_argument(
        "--min-bars",
        type=int,
        default=None,
        help="Warmup bars before signals fire. Default: 380 if tail<2500 else 900 (production-style).",
    )
    p.add_argument("--log", type=Path, default=DEFAULT_LIVE_LOG_PATH, help="JSONL log path")
    args = p.parse_args()

    tail = None if args.tail == 0 else args.tail
    min_bars = args.min_bars
    if min_bars is None:
        min_bars = 380 if (tail is not None and tail < 2500) else MIN_OHLC_BARS_WARMUP
    bot = LiveEURUSDBot(
        bar_source=CsvReplaySource(
            args.csv, symbol="EURUSD", timeframe="1h", tail_rows=tail
        ),
        log_path=args.log,
        min_bars=min_bars,
    )
    bot.run_replay_to_end()
    print(f"Replay complete. Log: {args.log}")
    print(f"Paper broker intents: {len(bot.broker.intents)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Bounded validation checks for daemon remediation hardening."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import DATA_DIR
from data.symbols import default_raw_csv_path
from live.broker.paper_broker import PaperBrokerClient
from live.ingestion import CsvReplaySource
from live.runner import LiveEURUSDBot


def _assert(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def _load_candles(tail: int = 2200) -> list:
    csv = default_raw_csv_path("EURUSD", DATA_DIR)
    src = CsvReplaySource(csv, symbol="EURUSD", timeframe="1h", tail_rows=tail)
    return list(src)


def validate_bootstrap_and_duplicates() -> dict:
    candles = _load_candles(tail=2200)
    broker = PaperBrokerClient(default_units=1000)
    bot = LiveEURUSDBot(
        bar_source=CsvReplaySource(default_raw_csv_path("EURUSD", DATA_DIR), tail_rows=10),
        min_bars=380,
        broker=broker,
        default_units=1000,
    )
    warm = 900
    for c in candles[:warm]:
        bot.on_warmup_bar(c)
    _assert(len(broker.intents) == 0, "Warmup should never place orders.")

    # Duplicate candle guard
    c0 = candles[warm]
    bot.on_closed_bar(c0, allow_trading=False)
    before_dupes = bot.metrics["duplicate_candles"]
    before_intents = len(broker.intents)
    bot.on_closed_bar(c0, allow_trading=True)
    _assert(
        bot.metrics["duplicate_candles"] == before_dupes + 1,
        "Duplicate candle counter did not increment.",
    )
    _assert(
        len(broker.intents) == before_intents,
        "Duplicate candle should not trigger order flow.",
    )
    return {
        "warmup_orders": len(broker.intents),
        "duplicate_candles": bot.metrics["duplicate_candles"],
    }


def validate_restart_recovery() -> dict:
    candles = _load_candles(tail=2600)
    broker_a = PaperBrokerClient(default_units=1000)
    bot_a = LiveEURUSDBot(
        bar_source=CsvReplaySource(default_raw_csv_path("EURUSD", DATA_DIR), tail_rows=10),
        min_bars=380,
        broker=broker_a,
        default_units=1000,
    )
    for c in candles[:1300]:
        bot_a.on_warmup_bar(c)
    for c in candles[1300:1700]:
        bot_a.on_closed_bar(c, allow_trading=True)
        if not bot_a.manager.is_flat:
            break

    snap = asdict(bot_a.snapshot_runtime())
    broker_b = PaperBrokerClient(default_units=1000)
    bot_b = LiveEURUSDBot(
        bar_source=CsvReplaySource(default_raw_csv_path("EURUSD", DATA_DIR), tail_rows=10),
        min_bars=380,
        broker=broker_b,
        default_units=1000,
    )
    restored = bot_b.restore_runtime(snap)
    _assert(restored, "Runtime snapshot restore failed.")
    _assert(
        str(bot_b.last_processed_ts) == str(bot_a.last_processed_ts),
        "last_processed_ts mismatch after restore.",
    )
    _assert(
        len(bot_b.engine.snapshot_fired_keys()) == len(bot_a.engine.snapshot_fired_keys()),
        "fired_keys mismatch after restore.",
    )
    if not bot_a.manager.is_flat:
        _assert(not bot_b.manager.is_flat, "Active position did not restore.")
    return {
        "restored": restored,
        "active_position_restored": (not bot_a.manager.is_flat) == (not bot_b.manager.is_flat),
    }


def validate_daemon_retry_and_watchdog() -> dict:
    with tempfile.TemporaryDirectory(prefix="alb_remediation_") as td:
        tdp = Path(td)
        port = 19099
        server_env = os.environ.copy()
        server_env["MOCK_OANDA_FAIL_FIRST"] = "2"
        server = subprocess.Popen(
            [sys.executable, "scripts/mock_oanda_candles_server.py", str(port)],
            cwd=PROJECT_ROOT,
            env=server_env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            time.sleep(0.5)
            env = os.environ.copy()
            env.update(
                {
                    "OANDA_REST_BASE": f"http://127.0.0.1:{port}",
                    "OANDA_API_TOKEN": "validate-token",
                    "OANDA_ACCOUNT_ID": "validate-account",
                    "OANDA_POLL_SECONDS": "1",
                    "OANDA_BOOTSTRAP_CANDLES": "120",
                    "LIVE_MIN_BARS": "80",
                    "OANDA_POLL_COUNT": "60",
                }
            )
            log_path = tdp / "daemon.jsonl"
            state_path = tdp / "state.json"
            watchdog_path = tdp / "watchdog.json"
            proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "live.run_daemon",
                    "--paper",
                    "--max-polls",
                    "2",
                    "--log",
                    str(log_path),
                    "--state",
                    str(state_path),
                    "--watchdog",
                    str(watchdog_path),
                ],
                cwd=PROJECT_ROOT,
                env=env,
                text=True,
                capture_output=True,
                timeout=120,
            )
            _assert(proc.returncode == 0, f"daemon failed: {proc.stdout}\n{proc.stderr}")
            lines = log_path.read_text(encoding="utf-8").splitlines()
            _assert(any('"kind": "poll_error"' in ln for ln in lines), "poll_error missing.")
            _assert(any('"kind": "heartbeat"' in ln for ln in lines), "heartbeat missing.")
            _assert(watchdog_path.exists(), "watchdog file not written.")
            _assert(state_path.exists(), "state file not written.")
            wd = json.loads(watchdog_path.read_text(encoding="utf-8"))
            _assert(wd.get("status") in {"ok", "degraded"}, "watchdog status missing.")
            return {
                "daemon_returncode": proc.returncode,
                "poll_error_logged": True,
                "heartbeat_logged": True,
                "watchdog_status": wd.get("status"),
            }
        finally:
            server.terminate()
            try:
                server.wait(timeout=2)
            except Exception:
                server.kill()


def main() -> int:
    out = {
        "bootstrap_duplicate": validate_bootstrap_and_duplicates(),
        "restart_recovery": validate_restart_recovery(),
        "daemon_retry_watchdog": validate_daemon_retry_and_watchdog(),
    }
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

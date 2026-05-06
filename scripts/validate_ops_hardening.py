"""Bounded validation for final ops hardening safeguards."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _assert(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _run_daemon(args: list[str], env: dict[str, str], timeout: int = 120) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "live.run_daemon", *args],
        cwd=PROJECT_ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=timeout,
    )


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="alb_ops_") as td:
        tdp = Path(td)
        port = 19111

        server = subprocess.Popen(
            [sys.executable, "scripts/mock_oanda_candles_server.py", str(port)],
            cwd=PROJECT_ROOT,
            env=os.environ.copy(),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            time.sleep(0.6)
            base_env = os.environ.copy()
            base_env.update(
                {
                    "OANDA_REST_BASE": f"http://127.0.0.1:{port}",
                    "OANDA_API_TOKEN": "ops-token",
                    "OANDA_ACCOUNT_ID": "ops-account",
                    "OANDA_POLL_SECONDS": "1",
                    "OANDA_BOOTSTRAP_CANDLES": "120",
                    "LIVE_MIN_BARS": "80",
                    "OANDA_POLL_COUNT": "60",
                    "LIVE_LOG_MAX_BYTES": "7000",
                    "LIVE_LOG_BACKUPS": "2",
                    "METRICS_HISTORY_MAX_BYTES": "1200",
                    "METRICS_HISTORY_BACKUPS": "2",
                }
            )

            log_path = tdp / "bot.jsonl"
            state_path = tdp / "state.json"
            watchdog_path = tdp / "watchdog.json"
            lock_path = tdp / "daemon.lock"
            metrics_path = tdp / "metrics.json"
            metrics_history = tdp / "metrics.jsonl"

            common_args = [
                "--paper",
                "--log",
                str(log_path),
                "--state",
                str(state_path),
                "--watchdog",
                str(watchdog_path),
                "--lock",
                str(lock_path),
                "--metrics",
                str(metrics_path),
                "--metrics-history",
                str(metrics_history),
            ]

            # Start daemon and validate singleton lock with second launch.
            p1 = subprocess.Popen(
                [sys.executable, "-m", "live.run_daemon", *common_args, "--max-polls", "8"],
                cwd=PROJECT_ROOT,
                env=base_env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            time.sleep(1.5)
            p2 = _run_daemon([*common_args, "--max-polls", "1"], base_env, timeout=60)
            _assert(p2.returncode != 0, "Second daemon launch should be rejected by lock.")
            _assert("already running" in (p2.stderr + p2.stdout), "Lock rejection message missing.")

            # Request graceful shutdown and verify state flush.
            p1.send_signal(signal.SIGTERM)
            out1, err1 = p1.communicate(timeout=30)
            _assert(p1.returncode == 0, f"Daemon did not exit cleanly: {out1}\n{err1}")
            _assert(state_path.exists(), "State file missing after graceful shutdown.")
            _assert(watchdog_path.exists(), "Watchdog file missing after graceful shutdown.")
            wd = _read_json(watchdog_path)
            _assert(wd.get("status") == "stopped", "Watchdog did not report stopped status.")
            st_before = _read_json(state_path)
            ts_before = st_before.get("runtime", {}).get("last_processed_ts")
            _assert(ts_before, "Missing last_processed_ts in persisted state.")

            # Restart and ensure state resumes/advances.
            p3 = _run_daemon([*common_args, "--max-polls", "2"], base_env, timeout=120)
            _assert(p3.returncode == 0, f"Restart daemon failed: {p3.stdout}\n{p3.stderr}")
            st_after = _read_json(state_path)
            ts_after = st_after.get("runtime", {}).get("last_processed_ts")
            _assert(ts_after, "Missing last_processed_ts after restart.")
            _assert(str(ts_after) >= str(ts_before), "Restart did not resume with durable state.")

            # Metrics update + rotation bounds
            _assert(metrics_path.exists(), "Metrics snapshot file missing.")
            m = _read_json(metrics_path)
            required = {
                "uptime_seconds",
                "poll_cycles",
                "poll_failures",
                "reconnect_count",
                "processed_candles",
                "detected_setups",
                "submitted_intents",
                "active_position",
            }
            _assert(required.issubset(set(m.keys())), "Metrics snapshot missing required fields.")
            history_files = [metrics_history, metrics_history.with_name(metrics_history.name + ".1")]
            _assert(any(p.exists() for p in history_files), "Metrics history not written/rotated.")
            _assert(log_path.exists(), "Primary log file missing.")
            # bounded logs: no index beyond configured backups=2
            _assert(
                not log_path.with_name(log_path.name + ".3").exists(),
                "Log backups exceed configured cap.",
            )
            _assert(
                not metrics_history.with_name(metrics_history.name + ".3").exists(),
                "Metrics history backups exceed configured cap.",
            )

            result = {
                "duplicate_daemon_rejected": True,
                "graceful_shutdown_state_written": True,
                "restart_resumed_state": True,
                "metrics_snapshot_written": True,
                "logs_bounded": True,
            }
            print(json.dumps(result, indent=2))
            return 0
        finally:
            server.terminate()
            try:
                server.wait(timeout=2)
            except Exception:
                server.kill()


if __name__ == "__main__":
    raise SystemExit(main())

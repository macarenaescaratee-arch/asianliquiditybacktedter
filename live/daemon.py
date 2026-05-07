"""
Continuous runner: OANDA H1 candle poll → ``LiveEURUSDBot`` + broker sync / fills.

Environment (see ``live.phase7_env.Phase7OandaConfig``):

- ``OANDA_API_TOKEN``, ``OANDA_ACCOUNT_ID``
- ``OANDA_ENV`` — ``practice`` (default) or ``live``
- ``OANDA_POLL_SECONDS`` — REST poll interval (default 15)
- ``OANDA_BOOTSTRAP_CANDLES`` — historical H1 bars to seed the buffer
- ``LIVE_MIN_BARS`` — signal engine warmup (default 900)
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path


from live.broker.base import BrokerClient
from live.broker.oanda_broker import OandaExecutionBroker
from live.broker.oanda_rest import OandaRestError
from live.broker.oanda_rest import OandaRestClient
from live.broker.paper_broker import PaperBrokerClient
from live.config import (
    DEFAULT_DAEMON_LOCK_PATH,
    DEFAULT_DAEMON_STATE_PATH,
    DEFAULT_LIVE_LOG_PATH,
    DEFAULT_METRICS_HISTORY_PATH,
    DEFAULT_METRICS_PATH,
    DEFAULT_WATCHDOG_PATH,
)
from live.feed.oanda_poll import bootstrap_closed_candles, poll_new_closed_candles
from live.ingestion import WebSocketBarSourceStub
from live.ops_runtime import SingletonPidLock
from live.phase7_env import Phase7OandaConfig
from live.runner import LiveEURUSDBot
from live.state_store import MetricsSnapshotStore, RuntimeStateStore, WatchdogStatusStore

LOG = logging.getLogger(__name__)


def _with_retries(func, *, max_backoff_s: float, base_wait_s: float, logger) -> object:
    failures = 0
    while True:
        try:
            return func()
        except (OandaRestError, TimeoutError, ConnectionError, RuntimeError) as exc:
            failures += 1
            wait_s = min(base_wait_s * (2 ** min(failures, 6)), max_backoff_s)
            logger.log(
                "poll_error",
                {
                    "error": str(exc),
                    "consecutive_failures": failures,
                    "backoff_seconds": wait_s,
                    "phase": "bootstrap",
                },
            )
            time.sleep(max(wait_s, 1.0))



def run_forever(
    *,
    paper: bool,
    log_path: Path | None = None,
    max_polls: int | None = None,
    state_path: Path | None = None,
    watchdog_path: Path | None = None,
    lock_path: Path | None = None,
    metrics_path: Path | None = None,
    metrics_history_path: Path | None = None,
) -> None:
    cfg = Phase7OandaConfig.from_environ()
    cfg.validate()

    rest = OandaRestClient(
        api_token=cfg.api_token,
        account_id=cfg.account_id,
        practice=cfg.practice,
    )

    if paper:
        broker: BrokerClient = PaperBrokerClient(default_units=cfg.default_units)
    else:
        broker = OandaExecutionBroker(
            rest,
            instrument=cfg.instrument,
            risk_fraction=cfg.risk_fraction,
            fallback_units=cfg.default_units,
        )

    bot_log_path = log_path or DEFAULT_LIVE_LOG_PATH
    bot = LiveEURUSDBot(
        bar_source=WebSocketBarSourceStub(symbol="EURUSD", timeframe="1h"),
        min_bars=cfg.live_min_bars,
        log_path=bot_log_path,
        broker=broker,
        default_units=cfg.default_units,
    )
    state_store = RuntimeStateStore(state_path or DEFAULT_DAEMON_STATE_PATH)
    watchdog = WatchdogStatusStore(watchdog_path or DEFAULT_WATCHDOG_PATH)
    lock = SingletonPidLock(lock_path or DEFAULT_DAEMON_LOCK_PATH)
    metrics_store = MetricsSnapshotStore(
        metrics_path or DEFAULT_METRICS_PATH,
        metrics_history_path or DEFAULT_METRICS_HISTORY_PATH,
        history_max_bytes=int(os.environ.get("METRICS_HISTORY_MAX_BYTES", str(5 * 1024 * 1024))),
        history_backups=int(os.environ.get("METRICS_HISTORY_BACKUPS", "6")),
    )
    startup_utc = datetime.now(timezone.utc)
lock_res = lock.acquire()
if not lock_res.acquired:
        raise RuntimeError(lock_res.message)
watchdog.update({"status": "starting", "pid": os.getpid()})
persisted = state_store.load()

    restored = False
    if persisted and isinstance(persisted, dict):
        broker.import_runtime_state(persisted.get("broker") or {})
        runtime = persisted.get("runtime")
        if isinstance(runtime, dict):
            restored = bot.restore_runtime(runtime)
            if restored:
                bot.logger.log(
                    "state_restore_ok",
                    {
                        "last_processed_ts": runtime.get("last_processed_ts"),
                        "state_path": str(state_store.path),
                    },
                )

    bootstrap = _with_retries(
        lambda: bootstrap_closed_candles(
            rest,
            cfg.instrument,
            count=cfg.candle_bootstrap,
        ),
        max_backoff_s=cfg.max_retry_backoff_seconds,
        base_wait_s=cfg.poll_seconds,
        logger=bot.logger,
    )
    if len(bootstrap) < cfg.live_min_bars:
        raise RuntimeError(
            f"Bootstrap returned {len(bootstrap)} candles; need >= {cfg.live_min_bars}."
        )

    last_ts = bot.last_processed_ts
    if not restored:
        for c in bootstrap:
            bot.on_warmup_bar(c)
            last_ts = c.ts
        bot.logger.log(
            "warmup_complete",
            {"bars": len(bootstrap), "last_ts": str(last_ts), "trading_enabled": True},
        )
    else:
        if last_ts is None:
            for c in bootstrap:
                bot.on_warmup_bar(c)
                last_ts = c.ts
        bot.logger.log(
            "warmup_skipped_restore",
            {"restored": True, "last_ts": str(last_ts)},
        )
    LOG.info(
        "Seeded %d H1 candles (instrument=%s); last_ts=%s",
        len(bootstrap),
        cfg.instrument,
        last_ts,
    )
    # Basic restart reconciliation: ensure restored position is represented in broker state.
    if restored and not bot.manager.is_flat and hasattr(broker, "_client"):
        try:
            open_trades = broker._client.get_open_trades()  # noqa: SLF001
            live_ids = {str(t.get("id")) for t in open_trades}
            pid = bot.manager.position.broker_trade_id if bot.manager.position else None
            if pid and not str(pid).startswith("paper_") and pid not in live_ids:
                bot.logger.log(
                    "reconcile_mismatch",
                    {
                        "issue": "restored_trade_missing_on_broker",
                        "broker_trade_id": pid,
                    },
                )
                bot.manager.position = None
        except Exception as exc:
            bot.logger.log("reconcile_error", {"error": str(exc)})

    poll_i = 0
    consecutive_failures = 0
    reconnect_count = 0
    metrics: dict[str, int] = {"poll_errors": 0, "poll_cycles": 0, "fill_events": 0}
    shutdown_reason = "unknown"
    try:
        while True:
            time.sleep(max(cfg.poll_seconds, 1.0))
            try:
                new_bars = poll_new_closed_candles(
                    rest,
                    cfg.instrument,
                    since_ts=last_ts,
                    count=cfg.poll_count,
                )
                acc = broker.sync_account()
                fills = broker.poll_fills()
                if consecutive_failures > 0:
                    reconnect_count += 1
                consecutive_failures = 0
            except (OandaRestError, TimeoutError, ConnectionError, RuntimeError) as exc:
                consecutive_failures += 1
                metrics["poll_errors"] += 1
                wait_s = min(
                    cfg.poll_seconds * (2 ** min(consecutive_failures, 6)),
                    cfg.max_retry_backoff_seconds,
                )
                bot.logger.log(
                    "poll_error",
                    {
                        "error": str(exc),
                        "consecutive_failures": consecutive_failures,
                        "backoff_seconds": wait_s,
                    },
                )
                if consecutive_failures >= cfg.alert_consecutive_failures:
                    bot.logger.log(
                        "alert",
                        {
                            "kind": "consecutive_poll_failures",
                            "count": consecutive_failures,
                            "error": str(exc),
                        },
                    )
                watchdog.update(
                    {
                        "status": "degraded",
                        "consecutive_failures": consecutive_failures,
                        "last_error": str(exc),
                        "poll_cycles": metrics["poll_cycles"],
                        "pid": os.getpid(),
                    }
                )
                time.sleep(max(wait_s, 1.0))
                continue

            metrics["poll_cycles"] += 1
            metrics["fill_events"] += len(fills)
            if acc:
                bot.logger.log(
                    "account_sync",
                    {
                        "balance": acc.balance,
                        "currency": acc.currency,
                        "margin_available": acc.margin_available,
                        "nav": acc.nav,
                        "margin_used": acc.margin_used,
                        "unrealized_pl": acc.unrealized_pl,
                    },
                )
            for f in fills:
                bot.logger.log(
                    "fill",
                    {
                        "transaction_id": f.transaction_id,
                        "instrument": f.instrument,
                        "time_utc": f.time_utc,
                        "units": f.units,
                        "price": f.price,
                        "pl": f.pl,
                        "financing": f.financing,
                    },
                )
            LOG.info(
                "poll: new_closed_bars=%d fill_events=%d balance=%s",
                len(new_bars),
                len(fills),
                f"{acc.balance:.2f}" if acc else "n/a",
            )
            for c in new_bars:
                bot.on_closed_bar(c)
                last_ts = c.ts
            runtime = asdict(bot.snapshot_runtime())
            state_store.save(
                {
                    "runtime": runtime,
                    "broker": broker.export_runtime_state(),
                    "metrics": {**metrics, **bot.metrics},
                    "mode": "paper" if paper else "live",
                    "instrument": cfg.instrument,
                    "log_path": str(bot_log_path),
                }
            )
            uptime_s = int((datetime.now(timezone.utc) - startup_utc).total_seconds())
            metrics_store.write(
                {
                    "mode": "paper" if paper else "live",
                    "pid": os.getpid(),
                    "uptime_seconds": uptime_s,
                    "poll_cycles": metrics["poll_cycles"],
                    "poll_failures": metrics["poll_errors"],
                    "reconnect_count": reconnect_count,
                    "processed_candles": bot.metrics["processed_candles"],
                    "duplicate_candles": bot.metrics["duplicate_candles"],
                    "detected_setups": bot.manager.metrics["detected_setups"],
                    "submitted_intents": bot.manager.metrics["submitted_intents"],
                    "active_position": not bot.manager.is_flat,
                    "last_processed_ts": str(last_ts) if last_ts is not None else None,
                }
            )
            watchdog.update(
                {
                    "status": "ok",
                    "pid": os.getpid(),
                    "uptime_seconds": uptime_s,
                    "last_processed_ts": str(last_ts) if last_ts is not None else None,
                    "poll_cycles": metrics["poll_cycles"],
                    "poll_errors": metrics["poll_errors"],
                    "duplicate_candles": bot.metrics["duplicate_candles"],
                    "open_position": not bot.manager.is_flat,
                }
            )
            bot.logger.log(
                "heartbeat",
                {
                    "poll_cycle": metrics["poll_cycles"],
                    "new_bars": len(new_bars),
                    "fill_events": len(fills),
                    "duplicate_candles": bot.metrics["duplicate_candles"],
                    "open_position": not bot.manager.is_flat,
                },
            )
            if max_polls is not None:
                poll_i += 1
                if poll_i >= max_polls:
                    shutdown_reason = f"max_polls={max_polls}"
                    LOG.info("Stopping after max_polls=%d (clean exit).", max_polls)
                    return
    except KeyboardInterrupt:
        shutdown_reason = "keyboard_interrupt"
        raise
    finally:
        runtime = asdict(bot.snapshot_runtime())
        state_store.save(
            {
                "runtime": runtime,
                "broker": broker.export_runtime_state(),
                "metrics": {**metrics, **bot.metrics},
                "mode": "paper" if paper else "live",
                "instrument": cfg.instrument,
                "log_path": str(bot_log_path),
                "shutdown_reason": shutdown_reason,
            }
        )
        uptime_s = int((datetime.now(timezone.utc) - startup_utc).total_seconds())
        watchdog.update(
            {
                "status": "stopped",
                "pid": os.getpid(),
                "uptime_seconds": uptime_s,
                "shutdown_reason": shutdown_reason,
            }
        )
        lock.release()

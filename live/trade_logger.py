"""Structured append-only trade / lifecycle logging."""

from __future__ import annotations

import os
from dataclasses import asdict, is_dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from live.config import DEFAULT_LIVE_LOG_PATH, LOG_DIR
from live.ops_runtime import RotatingJsonlWriter


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, datetime):
        return obj.replace(tzinfo=timezone.utc).isoformat()
    if isinstance(obj, date):
        return obj.isoformat()
    if hasattr(obj, "isoformat"):
        return obj.isoformat()
    if is_dataclass(obj):
        return {k: _json_safe(v) for k, v in asdict(obj).items()}
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(x) for x in obj]
    return obj


class TradeLogger:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path or DEFAULT_LIVE_LOG_PATH
        max_bytes = int(os.environ.get("LIVE_LOG_MAX_BYTES", str(20 * 1024 * 1024)))
        backups = int(os.environ.get("LIVE_LOG_BACKUPS", "8"))
        self._writer = RotatingJsonlWriter(self.path, max_bytes=max_bytes, backups=backups)

    def log(self, kind: str, payload: dict[str, Any]) -> None:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        rec = {
            "utc": datetime.now(timezone.utc).isoformat(),
            "kind": kind,
            **_json_safe(payload),
        }
        self._writer.append(rec)

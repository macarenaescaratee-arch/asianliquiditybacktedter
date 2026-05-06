"""Durable daemon runtime state and watchdog status persistence."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from live.ops_runtime import RotatingJsonlWriter

def _atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


class RuntimeStateStore:
    def __init__(self, path: Path) -> None:
        self.path = path

    def load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {}
        try:
            raw = self.path.read_text(encoding="utf-8")
            obj = json.loads(raw)
            return obj if isinstance(obj, dict) else {}
        except Exception:
            return {}

    def save(self, payload: dict[str, Any]) -> None:
        out = dict(payload)
        out["saved_utc"] = datetime.now(timezone.utc).isoformat()
        _atomic_write_json(self.path, out)


class WatchdogStatusStore:
    def __init__(self, path: Path) -> None:
        self.path = path

    def update(self, payload: dict[str, Any]) -> None:
        out = dict(payload)
        out["heartbeat_utc"] = datetime.now(timezone.utc).isoformat()
        _atomic_write_json(self.path, out)


class MetricsSnapshotStore:
    def __init__(
        self,
        snapshot_path: Path,
        history_path: Path,
        *,
        history_max_bytes: int,
        history_backups: int,
    ) -> None:
        self.snapshot_path = snapshot_path
        self._history = RotatingJsonlWriter(
            history_path,
            max_bytes=history_max_bytes,
            backups=history_backups,
        )

    def write(self, payload: dict[str, Any]) -> None:
        out = dict(payload)
        out["snapshot_utc"] = datetime.now(timezone.utc).isoformat()
        _atomic_write_json(self.snapshot_path, out)
        self._history.append(out)

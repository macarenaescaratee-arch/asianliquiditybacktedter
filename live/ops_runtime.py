"""Operational runtime helpers: singleton PID lock and rotating file writes."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _pid_is_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


@dataclass
class PidLockResult:
    acquired: bool
    existing_pid: int | None = None
    message: str = ""


class SingletonPidLock:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._held = False

    def acquire(self) -> PidLockResult:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            try:
                payload = json.loads(self.path.read_text(encoding="utf-8"))
                existing_pid = int(payload.get("pid", 0))
            except Exception:
                existing_pid = 0
            if _pid_is_alive(existing_pid):
                return PidLockResult(
                    acquired=False,
                    existing_pid=existing_pid,
                    message=f"daemon already running with pid={existing_pid}",
                )
            self.path.unlink(missing_ok=True)
        payload = {
            "pid": os.getpid(),
            "created_utc": datetime.now(timezone.utc).isoformat(),
        }
        self.path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        self._held = True
        return PidLockResult(acquired=True, existing_pid=None, message="acquired")

    def release(self) -> None:
        if self._held:
            self.path.unlink(missing_ok=True)
            self._held = False


class RotatingJsonlWriter:
    def __init__(self, path: Path, *, max_bytes: int, backups: int) -> None:
        self.path = path
        self.max_bytes = max(1024, int(max_bytes))
        self.backups = max(1, int(backups))

    def _rotate(self) -> None:
        if not self.path.exists():
            return
        if self.path.stat().st_size < self.max_bytes:
            return
        oldest = self.path.with_name(f"{self.path.name}.{self.backups}")
        oldest.unlink(missing_ok=True)
        for i in range(self.backups - 1, 0, -1):
            src = self.path.with_name(f"{self.path.name}.{i}")
            dst = self.path.with_name(f"{self.path.name}.{i+1}")
            if src.exists():
                src.replace(dst)
        self.path.replace(self.path.with_name(f"{self.path.name}.1"))

    def append(self, record: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._rotate()
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")


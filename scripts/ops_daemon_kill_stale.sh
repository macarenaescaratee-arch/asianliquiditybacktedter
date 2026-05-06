#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

LOCK_FILE="logs/eurusd_daemon.lock"
if [[ ! -f "$LOCK_FILE" ]]; then
  echo "no lock file found"
  exit 0
fi

PID="$(.venv/bin/python - <<'PY'
import json
from pathlib import Path
p=Path("logs/eurusd_daemon.lock")
try:
    d=json.loads(p.read_text(encoding="utf-8"))
    print(int(d.get("pid",0)))
except Exception:
    print(0)
PY
)"

if [[ "${PID}" -le 0 ]]; then
  rm -f "$LOCK_FILE"
  echo "removed invalid lock file"
  exit 0
fi

if kill -0 "$PID" 2>/dev/null; then
  kill -TERM "$PID"
  sleep 1
  if kill -0 "$PID" 2>/dev/null; then
    kill -KILL "$PID"
  fi
  echo "killed daemon pid=$PID"
else
  echo "pid $PID not running; removing stale lock"
fi

rm -f "$LOCK_FILE"

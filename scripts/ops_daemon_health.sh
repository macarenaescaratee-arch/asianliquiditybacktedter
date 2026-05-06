#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

WATCHDOG="logs/eurusd_daemon_watchdog.json"
if [[ ! -f "$WATCHDOG" ]]; then
  echo "missing watchdog file"
  exit 2
fi

.venv/bin/python - <<'PY'
import json,sys
from pathlib import Path
p=Path("logs/eurusd_daemon_watchdog.json")
d=json.loads(p.read_text(encoding="utf-8"))
status=d.get("status","unknown")
print(json.dumps({"status":status,"heartbeat_utc":d.get("heartbeat_utc"),"pid":d.get("pid")},indent=2))
if status not in {"ok","starting"}:
    sys.exit(1)
PY

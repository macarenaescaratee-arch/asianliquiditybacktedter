#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

for f in logs/eurusd_daemon_metrics.json logs/eurusd_daemon_watchdog.json logs/eurusd_daemon_state.json; do
  echo "==== $f ===="
  if [[ -f "$f" ]]; then
    .venv/bin/python - <<PY
import json
from pathlib import Path
p=Path("$f")
print(json.dumps(json.loads(p.read_text(encoding="utf-8")), indent=2))
PY
  else
    echo "(missing)"
  fi
done

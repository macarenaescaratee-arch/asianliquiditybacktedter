#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

mkdir -p logs

MODE="${1:---paper}"
if [[ "$MODE" != "--paper" && "$MODE" != "--live" ]]; then
  echo "usage: scripts/ops_daemon_start.sh [--paper|--live]"
  exit 1
fi

RUN_MODE_FLAG=""
if [[ "$MODE" == "--paper" ]]; then
  RUN_MODE_FLAG="--paper"
fi

nohup .venv/bin/python -m live.run_daemon \
  $RUN_MODE_FLAG \
  >> "logs/daemon_supervisor.log" 2>&1 &

echo "started daemon pid=$! mode=$MODE"

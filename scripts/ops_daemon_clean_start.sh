#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

scripts/ops_daemon_kill_stale.sh || true
rm -f logs/eurusd_daemon.lock

MODE="${1:---paper}"
scripts/ops_daemon_start.sh "$MODE"

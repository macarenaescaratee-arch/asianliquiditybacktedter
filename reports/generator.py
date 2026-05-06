"""
Report generation stubs.

Future use: take structured results from ``backtester`` and ``analytics``, write charts
(matplotlib for headless, plotly for interactive HTML), and assemble dated folders
under ``reports/`` for audit trails.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def ensure_reports_dir() -> Path:
    """Return absolute ``reports/`` path from config (creates directory if missing)."""
    from config import REPORTS_DIR

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    return REPORTS_DIR


def write_stub_report(payload: dict[str, Any], filename: str = "stub_run.json") -> Path:
    """
    Placeholder writer — replace with HTML/XLSX pipelines.

    Writes minimal JSON for wiring tests only; swap for real templates later.
    """
    import json

    out = ensure_reports_dir() / filename
    out.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return out

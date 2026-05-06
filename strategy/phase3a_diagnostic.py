"""
Phase 3A diagnostic: run Asian liquidity + MSS on all downloaded symbols and print samples.

Usage (project root)::

    python -m strategy.phase3a_diagnostic
"""

from __future__ import annotations

from dataclasses import dataclass

from data.loader import load_symbol_ohlcv_csv
from data.symbols import SUPPORTED_OHLC_SYMBOLS
from strategy.asian_liquidity_mss import (
    default_liquidity_mss_config,
    detect_asian_liquidity_mss,
)


@dataclass(slots=True)
class DiagnosticRow:
    date: str
    asset: str
    sweep_side: str
    mss_direction: str


def collect_setup_days(limit_total: int = 20) -> list[DiagnosticRow]:
    """Scan every supported symbol and gather setup candidates (most recent first)."""
    cfg = default_liquidity_mss_config()
    rows: list[DiagnosticRow] = []
    for sym in sorted(SUPPORTED_OHLC_SYMBOLS):
        ohlcv = load_symbol_ohlcv_csv(sym)
        daily = detect_asian_liquidity_mss(ohlcv, cfg, symbol=sym, eurusd_institutional=True)
        cand = daily[daily["tag"].str.contains("candidate", na=False)]
        for _, r in cand.iterrows():
            rows.append(
                DiagnosticRow(
                    date=str(r["session_date"]),
                    asset=sym,
                    sweep_side=str(r["sweep_side"]),
                    mss_direction=str(r["mss_direction"]),
                )
            )
    rows.sort(key=lambda x: x.date, reverse=True)
    return rows[:limit_total]


def print_diagnostic(rows: list[DiagnosticRow]) -> None:
    print("date       | asset   | asian sweep side | MSS direction")
    print("-" * 62)
    for r in rows:
        print(f"{r.date} | {r.asset:7} | {r.sweep_side:16} | {r.mss_direction}")
    print()
    print(f"(showing {len(rows)} of most recent setup candidates across all assets)")


def main() -> int:
    rows = collect_setup_days(20)
    if not rows:
        print("No setup candidates found across symbols (logic may be strict or data sparse).")
        return 1
    print_diagnostic(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

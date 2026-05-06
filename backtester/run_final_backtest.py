"""
Final integration: Asian MSS setups → trades → bar simulation → performance report.

Usage::

    python -m backtester.run_final_backtest
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from config import REPORTS_DIR
from data.loader import load_symbol_ohlcv_csv
from backtester.asian_mss_execution import (
    build_trade_setups,
    simulate_all,
    summarize_by_symbol,
)
from strategy.asian_liquidity_mss import default_liquidity_mss_config, detect_asian_liquidity_mss


ASSETS = ("EURUSD", "XAUUSD", "NAS100")


def _grand_total(results: list) -> dict:
    if not results:
        return {
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "timeouts": 0,
            "win_rate": 0.0,
            "avg_rr": 0.0,
            "expectancy_r": 0.0,
        }
    df = pd.DataFrame(
        {
            "outcome": [r.outcome for r in results],
            "rr": [r.realized_rr for r in results],
        }
    )
    n = len(df)
    return {
        "trades": n,
        "wins": int((df["outcome"] == "win").sum()),
        "losses": int((df["outcome"] == "loss").sum()),
        "timeouts": int((df["outcome"] == "timeout").sum()),
        "win_rate": float((df["outcome"] == "win").sum() / n) if n else 0.0,
        "avg_rr": float(df["rr"].mean()) if n else 0.0,
        "expectancy_r": float(df["rr"].mean()) if n else 0.0,
    }


def run() -> Path:
    cfg = default_liquidity_mss_config()
    all_results: list = []
    trade_rows: list[dict] = []
    lines: list[str] = []

    lines.append("AsianLiquidityBacktester — FINAL BACKTEST (Asian sweep + MSS)")
    lines.append("Assets: " + ", ".join(ASSETS))
    lines.append("RR target per winning trade: 1:2 (fixed). Stops: Asian extreme ± buffer vs sweep wick.")
    lines.append("")

    for sym in ASSETS:
        ohlcv = load_symbol_ohlcv_csv(sym)
        daily = detect_asian_liquidity_mss(ohlcv, cfg, symbol=sym, eurusd_institutional=True)
        setups = build_trade_setups(sym, ohlcv, cfg, daily)
        sims = simulate_all(ohlcv, setups)
        all_results.extend(sims)
        for r in sims:
            s = r.setup
            trade_rows.append(
                {
                    "symbol": s.symbol,
                    "session_date": s.session_date,
                    "direction": s.direction,
                    "entry_ts": s.entry_ts,
                    "entry": s.entry,
                    "stop": s.stop,
                    "take_profit": s.take_profit,
                    "rr_target": s.rr_target,
                    "risk_points": s.risk_points,
                    "risk_pips": s.risk_in_pips,
                    "asian_high": s.asian_high,
                    "asian_low": s.asian_low,
                    "outcome": r.outcome,
                    "realized_rr": r.realized_rr,
                    "exit_ts": r.exit_ts,
                    "exit_price": r.exit_price,
                }
            )
        lines.append(f"--- {sym} ---")
        lines.append(f"bars loaded: {len(ohlcv)}")
        lines.append(f"executable setups: {len(setups)}")
        lines.append(f"simulated trades: {len(sims)}")
        if setups:
            s0 = setups[0]
            lines.append(
                f"sample setup: {s0.session_date} {s0.direction} entry={s0.entry:.5f} SL={s0.stop:.5f} "
                f"TP={s0.take_profit:.5f} risk_pts={s0.risk_points:.5f} risk_pips≈{s0.risk_in_pips:.1f}"
            )
        lines.append("")

    per_sym = summarize_by_symbol(all_results)
    tot = _grand_total(all_results)

    lines.append("=== PER-ASSET PERFORMANCE ===")
    if per_sym.empty:
        lines.append("(no trades)")
    else:
        tbl = per_sym.copy()
        for col in ("win_rate",):
            tbl[col] = tbl[col].map(lambda x: f"{100 * x:.2f}%")
        for col in ("avg_rr", "expectancy_r"):
            tbl[col] = tbl[col].map(lambda x: f"{x:.3f}")
        lines.append(tbl.to_string(index=False))
        lines.append("")
        lines.append(f"Best asset (expectancy R):  {per_sym.attrs.get('best_asset', 'n/a')}")
        lines.append(f"Worst asset (expectancy R): {per_sym.attrs.get('worst_asset', 'n/a')}")

    lines.append("")
    lines.append("=== ALL ASSETS COMBINED ===")
    lines.append(
        f"total trades: {tot['trades']}\n"
        f"wins: {tot['wins']}\n"
        f"losses: {tot['losses']}\n"
        f"timeouts: {tot['timeouts']}\n"
        f"win rate: {100 * tot['win_rate']:.2f}%\n"
        f"average RR (realized R multiple): {tot['avg_rr']:.4f}\n"
        f"net expectancy (R per trade): {tot['expectancy_r']:.4f}"
    )

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REPORTS_DIR / "asian_mss_final_backtest.txt"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    csv_path = REPORTS_DIR / "asian_mss_trades.csv"
    if trade_rows:
        pd.DataFrame(trade_rows).to_csv(csv_path, index=False)

    print("\n".join(lines))
    print()
    print(f"Report written to: {out_path}")
    if trade_rows:
        print(f"Trade log CSV:      {csv_path}")
    return out_path


def main() -> int:
    run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

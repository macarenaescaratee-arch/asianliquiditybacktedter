"""
Phase 4: grid search execution / trade management for the fixed Phase 3 EURUSD setup set.

Does not change detection or institutional entry filters; only simulates post-entry paths.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from config import REPORTS_DIR
from data.loader import load_symbol_ohlcv_csv
from backtester.asian_mss_execution import build_trade_setups, simulate_all
from backtester.eurusd_phase4_execution import (
    ExecutionVariant,
    Phase4SimResult,
    aggregate_results,
    build_execution_variant_grid,
    compute_atr,
    simulate_variant,
)
from strategy.asian_liquidity_mss import default_liquidity_mss_config, detect_asian_liquidity_mss

SYMBOL = "EURUSD"
EXPECTED_TRADES = 46
REPORT_NAME = "PHASE4_EURUSD_QUANT_EXECUTION.txt"
CSV_NAME = "phase4_eurusd_best_trades.csv"


def _max_consecutive_losses(outcomes: list[str]) -> int:
    best = cur = 0
    for o in outcomes:
        if o == "loss":
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def _pick_best(df_rank: pd.DataFrame, variants: list[ExecutionVariant]) -> str:
    """Highest expectancy; tie-break prefers structural/hybrid SL (institutional), then total R."""
    df = df_rank.sort_values(
        ["expectancy_r", "total_r", "win_rate"],
        ascending=[False, False, False],
    )
    top_exp = float(df.iloc[0]["expectancy_r"])
    tied = df[np.isclose(df["expectancy_r"], top_exp, rtol=0, atol=1e-9)]
    # Lower is better: prefer structural Asian stop over fixed-pip when metrics tie
    sl_pref = {
        "structural": 0,
        "hybrid_struct_atr": 1,
        "structural_capped": 2,
        "atr": 3,
        "fixed_pip": 4,
    }
    best_name: str | None = None
    best_key: tuple[int, float, float] | None = None
    for _, row in tied.iterrows():
        name = str(row["variant"])
        v = next((x for x in variants if x.name == name), None)
        if v is None:
            continue
        key = (sl_pref.get(v.sl_mode, 99), float(row["total_r"]), float(row["win_rate"]))
        if best_key is None or key < best_key:
            best_key = key
            best_name = name
    return best_name or str(df.iloc[0]["variant"])


def run_phase4() -> tuple[Path, Path]:
    cfg = default_liquidity_mss_config()
    ohlcv = load_symbol_ohlcv_csv(SYMBOL)
    daily = detect_asian_liquidity_mss(ohlcv, cfg, symbol=SYMBOL, eurusd_institutional=True)
    setups = build_trade_setups(SYMBOL, ohlcv, cfg, daily)

    if len(setups) != EXPECTED_TRADES:
        raise RuntimeError(
            f"Phase 4 requires exactly {EXPECTED_TRADES} EURUSD institutional setups; got {len(setups)}. "
            "Detection/filters were not changed — verify data and filter wiring."
        )

    atr = compute_atr(ohlcv)
    variants = build_execution_variant_grid()

    sim_legacy = simulate_all(ohlcv, setups)
    legacy_total_r = sum(r.realized_rr for r in sim_legacy)
    legacy_exp = legacy_total_r / len(sim_legacy)

    rows_rank: list[dict] = []
    all_results: dict[str, list[Phase4SimResult]] = {}

    for v in variants:
        results = [simulate_variant(ohlcv, s, v, atr) for s in setups]
        all_results[v.name] = results
        agg = aggregate_results(results)
        total_r = float(sum(r.realized_r for r in results))
        rows_rank.append(
            {
                "variant": v.name,
                "n": int(agg["n"]),
                "expectancy_r": agg["exp"],
                "win_rate": agg["wr"],
                "avg_r": agg["avg_r"],
                "total_r": total_r,
            }
        )

    df_rank = pd.DataFrame(rows_rank)
    best_name = _pick_best(df_rank, variants)
    best_variant = next(v for v in variants if v.name == best_name)
    best_rows = all_results[best_name]

    baseline_key = "P3_NATIVE_spread0_capoff_200b"
    if baseline_key not in all_results:
        baseline_key = (
            "P3_PARITY_struct_2R"
            if "P3_PARITY_struct_2R" in all_results
            else next(iter(all_results.keys()))
        )
    baseline_rows = all_results[baseline_key]

    # --- Section 4: loser mitigation scan ---
    full_stop_idx = [
        i
        for i, r in enumerate(baseline_rows)
        if r.outcome == "loss" and r.realized_r <= -0.98
    ]
    mitigation_counts: dict[str, int] = {}
    for name, rs in all_results.items():
        if name == baseline_key:
            continue
        better = 0
        for i in full_stop_idx:
            if rs[i].realized_r > baseline_rows[i].realized_r + 1e-6:
                better += 1
        mitigation_counts[name] = better
    top_mitigation = sorted(mitigation_counts.items(), key=lambda x: -x[1])[:12]

    early_be_variants = [k for k in mitigation_counts if k.startswith("loser_BE") or "BE03" in k]
    timeout_variants = [k for k in mitigation_counts if "timeout" in k]

    # --- Section 5: winner expansion (MFE vs realized) ---
    leaked_r: list[tuple[int, float, float]] = []
    for i, r in enumerate(best_rows):
        if r.realized_r <= 0.05:
            continue
        gap = r.mfe_r - r.realized_r
        if gap >= 0.35:
            leaked_r.append((i, r.realized_r, gap))

    loser_analysis_lines = [
        f"Baseline for loss analysis: {baseline_key} (Phase 3–style path: native entry, 200 bars).",
        f"Full stop losses (about -1R each): {len(full_stop_idx)}",
        "Variants that most often improve on those full stops vs baseline (higher R on same trade index):",
    ]
    for nm, cnt in top_mitigation:
        loser_analysis_lines.append(f"  {nm}: {cnt} trades improved vs baseline full-stop indices")

    winner_analysis_lines = [
        f"Best variant `{best_name}`: trades with MFE R − realized R ≥ 0.35 (potential runner/trail upside): {len(leaked_r)}",
    ]
    for idx, rr, gap in sorted(leaked_r, key=lambda x: -x[2])[:15]:
        s = setups[idx]
        winner_analysis_lines.append(
            f"  idx={idx} {s.entry_ts} {s.direction}  realized={rr:.3f}  MFE_gap={gap:.3f}"
        )

    # --- Trade log CSV (best variant) ---
    trade_records = []
    for i, (s, r) in enumerate(zip(setups, best_rows)):
        trade_records.append(
            {
                "trade_index": i,
                "symbol": s.symbol,
                "session_date": str(s.session_date),
                "direction": s.direction,
                "entry_ts": s.entry_ts.isoformat(),
                "entry": s.entry,
                "setup_stop": s.stop,
                "setup_tp": s.take_profit,
                "phase4_variant": best_name,
                "realized_r": r.realized_r,
                "outcome": r.outcome,
                "exit_ts": r.exit_ts.isoformat() if r.exit_ts is not None else "",
                "mfe_r": r.mfe_r,
                "initial_risk_price_distance": r.initial_risk_price,
            }
        )
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = REPORTS_DIR / CSV_NAME
    pd.DataFrame(trade_records).to_csv(csv_path, index=False)

    best_total_r = sum(r.realized_r for r in best_rows)
    best_exp = best_total_r / len(best_rows)
    wins_best = sum(1 for r in best_rows if r.realized_r > 0.05)

    lines: list[str] = []
    lines.append("=" * 84)
    lines.append("PHASE 4 — EURUSD QUANT EXECUTION MODEL (46 Phase 3 institutional setups)")
    lines.append("=" * 84)
    lines.append("")
    lines.append("Scope: post-entry execution only. Candidate detection and Phase 3 entry filters unchanged.")
    lines.append("")
    lines.append("--- LEGACY ENGINE (asian_mss_execution._simulate_one), same 46 setups ---")
    lines.append(f"Expectancy (R/trade): {legacy_exp:.6f}")
    lines.append(f"Total R:            {legacy_total_r:.4f}")
    lines.append(
        f"Wins / Loss / TO:   {sum(1 for r in sim_legacy if r.outcome=='win')} / "
        f"{sum(1 for r in sim_legacy if r.outcome=='loss')} / "
        f"{sum(1 for r in sim_legacy if r.outcome=='timeout')}"
    )
    lines.append("")
    lines.append(f"--- BEST EXECUTION VARIANT: {best_name} ---")
    lines.append(repr(best_variant))
    lines.append(f"Expectancy (R/trade): {best_exp:.6f}")
    lines.append(f"Total R:            {best_total_r:.4f}")
    lines.append(f"Win rate (R>0.05): {100*wins_best/len(best_rows):.2f}%")
    lines.append(f"Max consecutive losses (outcome tag): {_max_consecutive_losses([r.outcome for r in best_rows])}")
    lines.append(f"Delta vs legacy total R: {best_total_r - legacy_total_r:+.4f}")
    lines.append("")
    if baseline_key in all_results:
        b = all_results[baseline_key]
        bt = sum(x.realized_r for x in b)
        lines.append(f"--- BASELINE PHASE4 SIM ({baseline_key}) ---")
        lines.append(f"Total R: {bt:.4f}  E[R]: {bt/len(b):.6f}")
        lines.append(f"Delta best vs Phase4-native baseline total R: {best_total_r - bt:+.4f}")
        lines.append("")

    lines.append("--- RANKED EXECUTION VARIANTS (by expectancy R/trade, descending) ---")
    show = df_rank.sort_values(["expectancy_r", "total_r"], ascending=[False, False])
    lines.append(show.to_string(index=False))
    lines.append("")

    lines.append("--- LOSER REDUCTION (full ~1R stops vs baseline indices) ---")
    lines.extend(loser_analysis_lines)
    lines.append("")
    lines.append("Sample early-BE / timeout variant improvement counts (same metric):")
    for label, keys in (
        ("early BE-style names", early_be_variants[:8]),
        ("timeout-style names", timeout_variants[:8]),
    ):
        lines.append(f"  {label}:")
        for k in keys:
            if k in mitigation_counts:
                lines.append(f"    {k}: {mitigation_counts[k]}")

    lines.append("")
    lines.append("--- WINNER EXPANSION (MFE vs realized on BEST variant) ---")
    lines.extend(winner_analysis_lines)

    lines.append("")
    lines.append("--- STANDALONE EURUSD PERFORMANCE (Phase 4 winning execution, 46 trades) ---")
    lines.append(f"Instrument: {SYMBOL}")
    lines.append(f"Trades: {len(best_rows)}")
    lines.append(f"Net R: {best_total_r:.4f}")
    lines.append(f"Expectancy: {best_exp:.6f} R/trade")
    lines.append(f"Average R: {np.mean([r.realized_r for r in best_rows]):.6f}")
    lines.append(f"Win rate (R > 0.05): {100*wins_best/len(best_rows):.2f}%")
    lines.append("")
    lines.append(
        "Reference: `FINAL_EURUSD_INSTITUTIONAL_REPORT.txt` lists legacy engine E[R] for the same 46 "
        f"setups (about 0.3085 R/trade). Phase 4 best model adds +{best_exp - legacy_exp:.4f} R/trade vs that snapshot."
    )
    lines.append("")
    lines.append(f"Trade log: {csv_path}")
    lines.append("")
    lines.append("=" * 84)
    lines.append("END OF PHASE 4 REPORT")
    lines.append("=" * 84)

    report_path = REPORTS_DIR / REPORT_NAME
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nSaved: {report_path}\nSaved: {csv_path}")

    return report_path, csv_path


def main() -> int:
    run_phase4()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

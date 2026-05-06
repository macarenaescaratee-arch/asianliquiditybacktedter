"""
Phase 5 — Monte Carlo, walk-forward / OOS, and friction stress for Phase 4 EURUSD execution.

Does not alter signal detection, filters, or the baseline execution policy; only evaluates
stability of outcomes under resampling, time splits, and transaction-cost stress.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.asian_mss_execution import TradeSetup
from backtester.eurusd_phase4_execution import (
    ExecutionVariant,
    phase4_eurusd_institutional_baseline,
    simulate_variant,
)


def max_drawdown_r(pnl_path: np.ndarray) -> float:
    """Max peak-to-trough drawdown in R on a cumulative PnL path (same units as pnl)."""
    if pnl_path.size == 0:
        return 0.0
    c = np.cumsum(pnl_path)
    peak = np.maximum.accumulate(c)
    return float(np.max(peak - c))


def _ruined(equity_path: np.ndarray) -> bool:
    return bool(equity_path.size and equity_path.min() <= 0.0)


def probability_of_ruin_monte_carlo(
    per_trade_r: np.ndarray,
    initial_capital_r: float,
    n_sims: int,
    rng: np.random.Generator,
    sample_mode: str = "bootstrap",
) -> float:
    """
    Fraction of paths where min(equity) <= 0, equity_0 = initial_capital_r,
    each trade adds one realized R to cumulative equity.

    ``sample_mode``: ``bootstrap`` (resample with replacement) or ``shuffle``
    (same multiset, random order — affects path risk only).
    """
    n = len(per_trade_r)
    if n == 0:
        return 0.0
    ruined = 0
    for _ in range(n_sims):
        if sample_mode == "bootstrap":
            draw = rng.choice(per_trade_r, size=n, replace=True)
        else:
            draw = rng.permutation(per_trade_r)
        eq = initial_capital_r + np.cumsum(draw)
        if _ruined(eq):
            ruined += 1
    return ruined / float(n_sims)


def monte_carlo_paths(
    per_trade_r: np.ndarray,
    n_sims: int,
    rng: np.random.Generator,
    sample_mode: str = "bootstrap",
) -> dict[str, np.ndarray]:
    """Returns arrays of total_R and max_DD per simulated path."""
    n = len(per_trade_r)
    totals = np.empty(n_sims)
    dds = np.empty(n_sims)
    for i in range(n_sims):
        if sample_mode == "bootstrap":
            draw = rng.choice(per_trade_r, size=n, replace=True)
        else:
            draw = rng.permutation(per_trade_r)
        totals[i] = float(draw.sum())
        dds[i] = max_drawdown_r(draw)
    return {"total_r": totals, "max_dd_r": dds}


def friction_grid_stats(
    ohlcv: pd.DataFrame,
    setups: list[TradeSetup],
    atr: pd.Series,
    base: ExecutionVariant,
    spreads: list[float],
    slippages: list[float],
) -> pd.DataFrame:
    rows = []
    for sp in spreads:
        for sl in slippages:
            v = dataclasses.replace(base, spread_pips=sp, slippage_pips=sl)
            rs = [simulate_variant(ohlcv, s, v, atr).realized_r for s in setups]
            arr = np.array(rs)
            rows.append(
                {
                    "spread_pips": sp,
                    "slippage_pips": sl,
                    "n": len(rs),
                    "total_r": float(arr.sum()),
                    "mean_r": float(arr.mean()),
                    "max_dd_r": max_drawdown_r(arr),
                }
            )
    return pd.DataFrame(rows)


def random_friction_monte_carlo(
    ohlcv: pd.DataFrame,
    setups: list[TradeSetup],
    atr: pd.Series,
    base: ExecutionVariant,
    n_sims: int,
    rng: np.random.Generator,
    spread_low: float = 0.2,
    spread_high: float = 1.2,
    slip_low: float = 0.0,
    slip_high: float = 0.5,
) -> dict[str, float]:
    """Full re-simulation with uniform random (spread, slippage) per draw."""
    totals: list[float] = []
    for _ in range(n_sims):
        sp = float(rng.uniform(spread_low, spread_high))
        sl = float(rng.uniform(slip_low, slip_high))
        v = dataclasses.replace(base, spread_pips=sp, slippage_pips=sl)
        rsum = sum(simulate_variant(ohlcv, s, v, atr).realized_r for s in setups)
        totals.append(rsum)
    t = np.array(totals)
    return {
        "mean_total_r": float(t.mean()),
        "p5_total_r": float(np.percentile(t, 5)),
        "p50_total_r": float(np.percentile(t, 50)),
        "p95_total_r": float(np.percentile(t, 95)),
        "p_lte_0": float((t <= 0).mean()),
    }


@dataclass(slots=True)
class WalkForwardSlice:
    label: str
    is_indices: list[int]
    oos_indices: list[int]
    is_mean_r: float
    oos_mean_r: float


def walk_forward_by_month(
    setups: list[TradeSetup],
    per_trade_r: np.ndarray,
    min_is_trades: int = 8,
) -> list[WalkForwardSlice]:
    """Expanding IS / rolling OOS: OOS = trades from month M onward (sorted chronologically)."""
    idx_ts = sorted(range(len(setups)), key=lambda i: setups[i].entry_ts)
    months: list[str] = []
    for i in idx_ts:
        ts = setups[i].entry_ts
        months.append(pd.Timestamp(ts).strftime("%Y-%m"))

    unique_months = sorted(set(months))
    slices: list[WalkForwardSlice] = []
    for m in unique_months:
        oos_idx = [idx_ts[j] for j in range(len(idx_ts)) if months[j] >= m]
        is_idx = [idx_ts[j] for j in range(len(idx_ts)) if months[j] < m]
        if len(is_idx) < min_is_trades or len(oos_idx) < 3:
            continue
        is_r = per_trade_r[is_idx]
        oos_r = per_trade_r[oos_idx]
        slices.append(
            WalkForwardSlice(
                label=f"OOS_from_{m}",
                is_indices=is_idx,
                oos_indices=oos_idx,
                is_mean_r=float(is_r.mean()),
                oos_mean_r=float(oos_r.mean()),
            )
        )
    return slices


def chronological_split_stats(
    per_trade_r_ordered: np.ndarray,
    frac_is: float,
) -> dict[str, float]:
    """Single chronological split: first ``frac_is`` fraction = IS, rest = OOS."""
    n = len(per_trade_r_ordered)
    k = int(max(1, min(n - 1, round(n * frac_is))))
    is_r = per_trade_r_ordered[:k]
    oos_r = per_trade_r_ordered[k:]
    return {
        "is_n": float(len(is_r)),
        "oos_n": float(len(oos_r)),
        "is_mean_r": float(is_r.mean()),
        "oos_mean_r": float(oos_r.mean()),
        "oos_minus_is": float(oos_r.mean() - is_r.mean()),
    }


def phase5_report_lines(
    n_trades: int,
    baseline_total_r: float,
    baseline_mean_r: float,
    baseline_dd: float,
    friction_df: pd.DataFrame,
    mc_boot: dict[str, np.ndarray],
    mc_shuffle: dict[str, np.ndarray],
    ruin_boot: dict[float, float],
    ruin_shuffle: dict[float, float],
    wf_slices: list[WalkForwardSlice],
    split_50: dict[str, float],
    split_60: dict[str, float],
    random_friction: dict[str, float],
    rng_seed: int,
    n_friction_mc: int,
) -> list[str]:
    lines: list[str] = []
    lines.append("=" * 84)
    lines.append("PHASE 5 — MONTE CARLO ROBUSTNESS / WALK-FORWARD / OVERFITTING CHECK (EURUSD)")
    lines.append("=" * 84)
    lines.append("")
    lines.append(f"Baseline: Phase 4 institutional execution ({phase4_eurusd_institutional_baseline().name})")
    lines.append(f"RNG seed: {rng_seed}")
    lines.append("")
    lines.append("--- POINT ESTIMATE (full sample, chronological order) ---")
    lines.append(f"Trades: {n_trades}")
    lines.append(f"Total R: {baseline_total_r:.4f}")
    lines.append(f"Mean R / trade: {baseline_mean_r:.6f}")
    lines.append(f"Max drawdown (ordered path): {baseline_dd:.4f} R")
    lines.append("")
    lines.append("--- SPREAD / SLIPPAGE STRESS (deterministic grid, full bar resimulation) ---")
    lines.append(
        "Note: on hourly bars and capped structural risk, moderate spread-only changes often leave the same "
        "discrete outcomes; large spreads (see grid tail) and slippage move totals."
    )
    lines.append(friction_df.to_string(index=False))
    lines.append("")
    lines.append(
        f"--- RANDOM FRICTION (uniform spread and slippage, full resimulation, {n_friction_mc} draws) ---"
    )
    for k, v in random_friction.items():
        lines.append(f"{k}: {v:.6f}")
    lines.append("")
    lines.append("--- MONTE CARLO: BOOTSTRAP (resample trades with replacement, n=15000) ---")
    for name, arr in mc_boot.items():
        lines.append(
            f"{name}: mean={arr.mean():.4f}  p5={np.percentile(arr,5):.4f}  "
            f"p50={np.percentile(arr,50):.4f}  p95={np.percentile(arr,95):.4f}"
        )
    lines.append(f"P(total R <= 0): {(mc_boot['total_r'] <= 0).mean():.4f}")
    lines.append("")
    lines.append("--- MONTE CARLO: SHUFFLE (same outcomes, random order — path / DD risk) ---")
    lines.append(
        f"total_R is invariant under permutation (always {baseline_total_r:.4f}); "
        "only drawdown and ruin depend on order."
    )
    lines.append(
        f"max_dd_r: mean={mc_shuffle['max_dd_r'].mean():.4f}  p5={np.percentile(mc_shuffle['max_dd_r'],5):.4f}  "
        f"p50={np.percentile(mc_shuffle['max_dd_r'],50):.4f}  p95={np.percentile(mc_shuffle['max_dd_r'],95):.4f}"
    )
    lines.append("")
    lines.append("--- PROBABILITY OF RUIN (min cumulative equity <= 0; start at K x 1R bankroll) ---")
    lines.append("Bootstrap resampling:")
    for k, p in sorted(ruin_boot.items()):
        lines.append(f"  Initial capital {k:.0f} R: P(ruin) = {p:.4f}")
    lines.append("Shuffle (order risk only):")
    for k, p in sorted(ruin_shuffle.items()):
        lines.append(f"  Initial capital {k:.0f} R: P(ruin) = {p:.4f}")
    lines.append("")
    lines.append("--- WALK-FORWARD / OOS (expanding IS by calendar month) ---")
    if not wf_slices:
        lines.append("(no valid month splits — insufficient trades per segment)")
    else:
        for s in wf_slices[:24]:
            lines.append(
                f"{s.label}: IS n={len(s.is_indices)} mean R={s.is_mean_r:.4f} | "
                f"OOS n={len(s.oos_indices)} mean R={s.oos_mean_r:.4f}"
            )
        if len(wf_slices) > 24:
            lines.append(f"... ({len(wf_slices) - 24} more slices omitted)")
    lines.append("")
    lines.append("--- OVERFITTING / STABILITY (single chronological splits, mean R) ---")
    lines.append(f"50/50 split: {split_50}")
    lines.append(f"60/40 IS/OOS split: {split_60}")
    lines.append("")
    lines.append("Interpretation: large gaps IS vs OOS mean R or high P(ruin) at realistic capital suggest fragility.")
    lines.append("Small sample (46 trades): use ranges and percentiles, not point significance.")
    lines.append("")
    lines.append("=" * 84)
    lines.append("END OF PHASE 5 REPORT")
    lines.append("=" * 84)
    return lines

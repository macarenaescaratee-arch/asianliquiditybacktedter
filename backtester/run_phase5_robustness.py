"""
Phase 5 runner: Monte Carlo, walk-forward OOS, friction stress, probability of ruin.

Baseline execution: Phase 4 institutional model (see ``phase4_eurusd_institutional_baseline``).
"""

from __future__ import annotations

import numpy as np

from config import REPORTS_DIR
from data.loader import load_symbol_ohlcv_csv
from backtester.asian_mss_execution import build_trade_setups
from backtester.eurusd_phase5_robustness import (
    chronological_split_stats,
    friction_grid_stats,
    max_drawdown_r,
    monte_carlo_paths,
    phase5_report_lines,
    probability_of_ruin_monte_carlo,
    random_friction_monte_carlo,
    walk_forward_by_month,
)
from backtester.eurusd_phase4_execution import compute_atr, phase4_eurusd_institutional_baseline, simulate_variant
from strategy.asian_liquidity_mss import default_liquidity_mss_config, detect_asian_liquidity_mss

SYMBOL = "EURUSD"
EXPECTED_TRADES = 46
REPORT_NAME = "PHASE5_EURUSD_ROBUSTNESS.txt"
RNG_SEED = 42
N_MC = 15_000
N_FRICTION_MC = 800


def main() -> int:
    rng = np.random.default_rng(RNG_SEED)
    cfg = default_liquidity_mss_config()
    ohlcv = load_symbol_ohlcv_csv(SYMBOL)
    daily = detect_asian_liquidity_mss(ohlcv, cfg, symbol=SYMBOL, eurusd_institutional=True)
    setups = build_trade_setups(SYMBOL, ohlcv, cfg, daily)
    if len(setups) != EXPECTED_TRADES:
        raise RuntimeError(f"Expected {EXPECTED_TRADES} EURUSD setups, got {len(setups)}")

    base = phase4_eurusd_institutional_baseline()
    atr = compute_atr(ohlcv)

    results = [simulate_variant(ohlcv, s, base, atr) for s in setups]
    order = sorted(range(len(setups)), key=lambda i: setups[i].entry_ts)
    r_ordered = np.array([results[i].realized_r for i in order])

    baseline_total = float(r_ordered.sum())
    baseline_mean = float(r_ordered.mean())
    baseline_dd = max_drawdown_r(r_ordered)

    spreads = [0.15, 0.25, 0.35, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
    slips = [0.0, 0.05, 0.1, 0.2, 0.35, 0.5]
    friction_df = friction_grid_stats(ohlcv, setups, atr, base, spreads, slips)

    rf = random_friction_monte_carlo(
        ohlcv,
        setups,
        atr,
        base,
        n_sims=N_FRICTION_MC,
        rng=rng,
        spread_low=0.2,
        spread_high=1.2,
        slip_low=0.0,
        slip_high=0.5,
    )

    mc_boot = monte_carlo_paths(r_ordered, N_MC, rng, sample_mode="bootstrap")
    mc_shuf = monte_carlo_paths(r_ordered, N_MC, rng, sample_mode="shuffle")

    capitals = [5.0, 10.0, 15.0, 20.0, 30.0]
    ruin_boot = {k: probability_of_ruin_monte_carlo(r_ordered, k, N_MC, rng, "bootstrap") for k in capitals}
    ruin_shuf = {k: probability_of_ruin_monte_carlo(r_ordered, k, N_MC, rng, "shuffle") for k in capitals}

    per_idx = np.array([results[i].realized_r for i in range(len(setups))])
    wf = walk_forward_by_month(setups, per_idx, min_is_trades=8)

    split_50 = chronological_split_stats(r_ordered, 0.5)
    split_60 = chronological_split_stats(r_ordered, 0.6)

    lines = phase5_report_lines(
        n_trades=len(r_ordered),
        baseline_total_r=baseline_total,
        baseline_mean_r=baseline_mean,
        baseline_dd=baseline_dd,
        friction_df=friction_df,
        mc_boot=mc_boot,
        mc_shuffle=mc_shuf,
        ruin_boot=ruin_boot,
        ruin_shuffle=ruin_shuf,
        wf_slices=wf,
        split_50=split_50,
        split_60=split_60,
        random_friction=rf,
        rng_seed=RNG_SEED,
        n_friction_mc=N_FRICTION_MC,
    )

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REPORTS_DIR / REPORT_NAME
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nSaved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

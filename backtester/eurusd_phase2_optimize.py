"""
PHASE 2 — EURUSD-only optimization for Asian Liquidity Sweep + MSS.

Analyzes losing vs winning trades, grid-searches quality filters, and reports the best
configuration by expectancy (with minimum trade count guardrail).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import time
import json
from itertools import product
from typing import Any, Literal

import numpy as np
import pandas as pd

from config import REPORTS_DIR
from data.loader import load_symbol_ohlcv_csv
from backtester.asian_mss_execution import (
    TradeSetup,
    _pip_size,
    _seg_after,
    build_trade_setups,
    simulate_all,
)
from strategy.asian_liquidity_mss import default_liquidity_mss_config, detect_asian_liquidity_mss
from strategy.mss_audit_report import _london_ny_flags

SYMBOL = "EURUSD"
MIN_TRADES_FOR_RANKING = 35


def _atr14(ohlcv: pd.DataFrame) -> pd.Series:
    h, l, c = ohlcv["high"], ohlcv["low"], ohlcv["close"]
    pc = c.shift(1)
    tr = pd.concat([(h - l).abs(), (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.rolling(14, min_periods=5).mean()


def _sweep_session_bucket(ts: pd.Timestamp) -> Literal["London", "NY", "overlap", "other"]:
    l_ok, n_ok = _london_ny_flags(ts)
    if l_ok and n_ok:
        return "overlap"
    if l_ok:
        return "London"
    if n_ok:
        return "NY"
    return "other"


def _first_displacement_metrics(
    seg_after: pd.DataFrame,
    cfg: LiquidityMSSConfig,
    sweep_ts: pd.Timestamp,
    bullish: bool,
) -> tuple[float | None, float | None, float | None]:
    """
    Returns (body_ratio, range_pips, range_to_atr) for first qualifying displacement bar, or Nones.
    """
    from strategy.asian_liquidity_mss import (
        _is_bearish_displacement,
        _is_bullish_displacement,
        _segment_range_ref,
    )

    if seg_after.empty:
        return None, None, None
    ref = _segment_range_ref(seg_after)
    pip = _pip_size(SYMBOL)
    for ts, row in seg_after.iterrows():
        if ts <= sweep_ts:
            continue
        ok = (
            _is_bullish_displacement(row, ref, cfg.displacement_body_ratio, cfg.displacement_min_range_ratio)
            if bullish
            else _is_bearish_displacement(row, ref, cfg.displacement_body_ratio, cfg.displacement_min_range_ratio)
        )
        if not ok:
            continue
        o, h, l, cl = float(row["open"]), float(row["high"]), float(row["low"]), float(row["close"])
        rng = h - l
        if rng <= 0:
            return None, None, None
        body = abs(cl - o)
        br = body / rng
        rp = rng / pip
        return br, rp, None  # atr ratio filled outside
    return None, None, None


def _extract_row_features(
    ohlcv: pd.DataFrame,
    cfg: LiquidityMSSConfig,
    atr: pd.Series,
    s: TradeSetup,
) -> dict[str, Any]:
    """Per-setup diagnostics for optimization (EURUSD)."""
    pip = _pip_size(SYMBOL)
    sweep_row = ohlcv.loc[s.sweep_ts]
    hi_s, lo_s = float(sweep_row["high"]), float(sweep_row["low"])

    if s.direction == "long":
        sweep_pen_pips = max(0.0, (s.asian_low - lo_s) / pip)
        bullish_disp = True
    else:
        sweep_pen_pips = max(0.0, (hi_s - s.asian_high) / pip)
        bullish_disp = False

    seg_after = _seg_after(ohlcv, s.sweep_ts, cfg)
    br, rp, _ = _first_displacement_metrics(seg_after, cfg, s.sweep_ts, bullish_disp)
    atr_s = float(atr.reindex([s.sweep_ts]).ffill().bfill().iloc[0]) if len(atr) else np.nan
    disp_rng_pip = rp
    rng_atr = (disp_rng_pip * pip) / atr_s if atr_s and atr_s > 0 and disp_rng_pip is not None else np.nan

    hrs = (s.confirm_ts - s.sweep_ts).total_seconds() / 3600.0

    return {
        "session_date": s.session_date,
        "direction": s.direction,
        "sweep_session": _sweep_session_bucket(s.sweep_ts),
        "sweep_penetration_pips": sweep_pen_pips,
        "disp_body_ratio": br,
        "disp_range_pips": rp,
        "disp_range_over_atr": rng_atr,
        "hours_sweep_to_confirm": hrs,
        "risk_pips": s.risk_in_pips,
    }


def _loser_winner_analysis(df: pd.DataFrame) -> str:
    """Plain-English summary of why losers differ from winners."""
    lines: list[str] = []
    lines.append("=== EURUSD LOSS DRIVER ANALYSIS (baseline unfiltered) ===")
    if df.empty:
        return "\n".join(lines)
    w = df[df["outcome"] == "win"]
    l = df[df["outcome"] == "loss"]
    t = df[df["outcome"] == "timeout"]
    lines.append(f"Counts — wins: {len(w)}, losses: {len(l)}, timeouts: {len(t)}")
    lines.append("")
    lines.append("Mean feature comparison (wins vs losses):")
    for col in [
        "sweep_penetration_pips",
        "disp_body_ratio",
        "disp_range_pips",
        "disp_range_over_atr",
        "hours_sweep_to_confirm",
        "risk_pips",
    ]:
        if col not in df.columns:
            continue
        mw = float(w[col].mean()) if len(w) else float("nan")
        ml = float(l[col].mean()) if len(l) else float("nan")
        lines.append(f"  {col:28}  wins={mw:8.3f}  losses={ml:8.3f}")
    lines.append("")
    lines.append("Session mix among LOSSES (share of losing trades):")
    if len(l):
        vc = l["sweep_session"].value_counts(normalize=True)
        for k, v in vc.items():
            lines.append(f"  {k}: {100 * float(v):.1f}%")
    lines.append("")
    lines.append("Timeouts vs sweep depth (mean penetration pips):")
    lines.append(f"  timeouts: {float(t['sweep_penetration_pips'].mean()):.3f}  wins: {float(w['sweep_penetration_pips'].mean()):.3f}")
    lines.append("")
    lines.append(
        "Interpretation (automated):\n"
        "  • If losses cluster in NY-only or show weaker mean displacement body/range, tighten session or displacement.\n"
        "  • If timeouts show shallow sweeps, require minimum penetration pips.\n"
        "  • If losses confirm faster/slower than wins, narrow sweep→confirm hours.\n"
    )
    return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class OptFilterPack:
    """Stricter post-hoc filters on already-valid MSS setups."""

    name: str
    sweep_session_mode: Literal["any", "london_only", "ny_only", "overlap_only", "no_overlap"]
    min_sweep_penetration_pips: float
    min_disp_body_ratio: float
    min_disp_range_over_atr: float
    min_hours_sweep_to_confirm: float
    max_hours_sweep_to_confirm: float
    min_risk_pips: float
    max_risk_pips: float

    def mask(self, feat: pd.DataFrame) -> pd.Series:
        s = feat["sweep_session"]
        if self.sweep_session_mode == "london_only":
            m_sess = (s == "London") | (s == "overlap")
        elif self.sweep_session_mode == "ny_only":
            m_sess = (s == "NY") | (s == "overlap")
        elif self.sweep_session_mode == "overlap_only":
            m_sess = s == "overlap"
        elif self.sweep_session_mode == "no_overlap":
            m_sess = s == "London"
        else:
            m_sess = pd.Series(True, index=feat.index)

        m_pen = feat["sweep_penetration_pips"] >= self.min_sweep_penetration_pips
        m_body = feat["disp_body_ratio"].fillna(0) >= self.min_disp_body_ratio
        if self.min_disp_range_over_atr > 0:
            m_atr = feat["disp_range_over_atr"].fillna(0) >= self.min_disp_range_over_atr
        else:
            m_atr = pd.Series(True, index=feat.index)
        m_h1 = feat["hours_sweep_to_confirm"] >= self.min_hours_sweep_to_confirm
        m_h2 = feat["hours_sweep_to_confirm"] <= self.max_hours_sweep_to_confirm
        m_r1 = feat["risk_pips"] >= self.min_risk_pips
        m_r2 = feat["risk_pips"] <= self.max_risk_pips
        return m_sess & m_pen & m_body & m_atr & m_h1 & m_h2 & m_r1 & m_r2


def _metrics(results: list) -> dict[str, float]:
    if not results:
        return {"n": 0, "wr": 0.0, "avg_rr": 0.0, "exp": 0.0}
    d = pd.DataFrame({"o": [r.outcome for r in results], "rr": [r.realized_rr for r in results]})
    n = len(d)
    wr = float((d["o"] == "win").sum() / n)
    avg_rr = float(d["rr"].mean())
    return {"n": n, "wr": wr, "avg_rr": avg_rr, "exp": avg_rr}


def _build_filter_grid() -> list[OptFilterPack]:
    """Stricter configurations (reduced grid for runtime; still multi-dimensional)."""
    packs: list[OptFilterPack] = []
    modes: list[Any] = ["any", "london_only", "ny_only", "no_overlap"]
    pen = [0.0, 2.0, 4.0]
    body = [0.58, 0.65, 0.72]
    atrm = [0.0, 0.35]
    hmin = [0.0, 1.5]
    hmax = [48.0, 28.0]
    rmin = [8.0]
    rmax = [150.0]
    idx = 0
    for mode, p, b, a, hm, hx, r1, r2 in product(modes, pen, body, atrm, hmin, hmax, rmin, rmax):
        if hm > hx:
            continue
        idx += 1
        packs.append(
            OptFilterPack(
                name=f"cfg_{idx:04d}",
                sweep_session_mode=mode,
                min_sweep_penetration_pips=p,
                min_disp_body_ratio=b,
                min_disp_range_over_atr=a,
                min_hours_sweep_to_confirm=hm,
                max_hours_sweep_to_confirm=hx,
                min_risk_pips=r1,
                max_risk_pips=r2,
            )
        )
    return packs


def run() -> OptFilterPack | None:
    cfg = default_liquidity_mss_config()
    ohlcv = load_symbol_ohlcv_csv(SYMBOL)
    daily = detect_asian_liquidity_mss(ohlcv, cfg)
    setups = build_trade_setups(SYMBOL, ohlcv, cfg, daily)
    atr = _atr14(ohlcv)

    feats = [_extract_row_features(ohlcv, cfg, atr, s) for s in setups]
    feat_df = pd.DataFrame(feats)
    base_res = simulate_all(ohlcv, setups)
    feat_df["outcome"] = [r.outcome for r in base_res]
    feat_df["realized_rr"] = [r.realized_rr for r in base_res]

    analysis_txt = _loser_winner_analysis(feat_df)

    rows_out: list[dict[str, Any]] = []
    baseline = _metrics(base_res)
    rows_out.append(
        {
            "name": "BASELINE_UNFILTERED",
            **baseline,
        }
    )

    best_score: tuple[float, float, int] = (-1e18, -1.0, -1)
    best_pack: OptFilterPack | None = None
    best_met: dict[str, float] = {}
    grid = _build_filter_grid()

    for pack in grid:
        m = pack.mask(feat_df)
        sub = [s for s, ok in zip(setups, m) if ok]
        if len(sub) < MIN_TRADES_FOR_RANKING:
            continue
        res = simulate_all(ohlcv, sub)
        met = _metrics(res)
        rows_out.append({"name": pack.name, **met})
        cand = (met["exp"], met["wr"], int(met["n"]))
        if cand > best_score:
            best_score = cand
            best_pack = pack
            best_met = met

    # If grid failed min trades, relax and try curated strong filters
    if best_pack is None:
        curated = [
            OptFilterPack("cur_london_deep", "london_only", 2.0, 0.62, 0.3, 1.0, 36.0, 8.0, 150.0),
            OptFilterPack("cur_london_body", "london_only", 1.0, 0.68, 0.0, 0.5, 48.0, 6.0, 180.0),
            OptFilterPack("cur_ny_deep", "ny_only", 2.5, 0.60, 0.25, 0.0, 24.0, 8.0, 130.0),
            OptFilterPack("cur_overlap", "overlap_only", 1.5, 0.65, 0.35, 1.0, 30.0, 10.0, 140.0),
        ]
        for pack in curated:
            m = pack.mask(feat_df)
            sub = [s for s, ok in zip(setups, m) if ok]
            if len(sub) < 25:
                continue
            res = simulate_all(ohlcv, sub)
            met = _metrics(res)
            rows_out.append({"name": pack.name, **met})
            cand = (met["exp"], met["wr"], int(met["n"]))
            if cand > best_score:
                best_score = cand
                best_pack = pack
                best_met = met

    res_df = pd.DataFrame(rows_out)
    res_df = res_df.sort_values("exp", ascending=False).reset_index(drop=True)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / "eurusd_phase2_optimization.txt"
    lines = [
        "EURUSD — PHASE 2 OPTIMIZATION REPORT",
        "",
        analysis_txt,
        "",
        "=== VARIANT COMPARISON (sorted by expectancy) ===",
        res_df.head(40).to_string(index=False),
        "",
        "=== BEST CONFIGURATION ===",
    ]
    if best_pack is not None:
        lines.append(f"Expectancy (R/trade): {best_met['exp']:.4f}")
        lines.append(f"Trades: {int(best_met['n'])}  Win rate: {100*best_met['wr']:.2f}%  Avg RR: {best_met['avg_rr']:.4f}")
        lines.append(f"Best filter pack: {best_pack}")
    else:
        lines.append("No configuration met minimum trade count.")

    report_path.write_text("\n".join(lines), encoding="utf-8")

    if best_pack is not None:
        js = REPORTS_DIR / "eurusd_phase2_best_filter.json"
        js.write_text(
            json.dumps(
                {
                    "sweep_session_mode": best_pack.sweep_session_mode,
                    "min_sweep_penetration_pips": best_pack.min_sweep_penetration_pips,
                    "min_disp_body_ratio": best_pack.min_disp_body_ratio,
                    "min_disp_range_over_atr": best_pack.min_disp_range_over_atr,
                    "min_hours_sweep_to_confirm": best_pack.min_hours_sweep_to_confirm,
                    "max_hours_sweep_to_confirm": best_pack.max_hours_sweep_to_confirm,
                    "min_risk_pips": best_pack.min_risk_pips,
                    "max_risk_pips": best_pack.max_risk_pips,
                    "metrics": {k: float(v) for k, v in best_met.items()},
                    "baseline_metrics": {k: float(v) for k, v in baseline.items()},
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    print(analysis_txt)
    print()
    print("=== TOP OPTIMIZATION RESULTS (head) ===")
    print(res_df.head(25).to_string(index=False))
    print()
    if best_pack is not None:
        print("=== BEST CONFIGURATION ===")
        print(best_pack)
        print("metrics:", best_met)
    print()
    print(f"Full report: {report_path}")
    if best_pack is not None:
        print(f"Best filter JSON: {REPORTS_DIR / 'eurusd_phase2_best_filter.json'}")
    return best_pack


def main() -> int:
    run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

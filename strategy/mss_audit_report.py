"""
Human-readable audit for Asian Liquidity Sweep + MSS (read-only diagnostics).

Does not change detection logic; reproduces sweep timestamps and explains outcomes
using the same helpers as ``asian_liquidity_mss``.
"""

from __future__ import annotations

from datetime import datetime, time
import pandas as pd

from data.loader import load_symbol_ohlcv_csv
from data.symbols import SUPPORTED_OHLC_SYMBOLS
from strategy.asian_liquidity_mss import (
    LiquidityMSSConfig,
    annotate_tokyo_calendar,
    build_london_ny_killzone_mask,
    default_liquidity_mss_config,
    detect_asian_liquidity_mss,
    sweep_session_mask,
    _bearish_mss_after_sweep_up,
    _bullish_mss_after_sweep_down,
    _first_liquidity_sweep,
    _is_bearish_displacement,
    _is_bullish_displacement,
    _segment_range_ref,
)


def _london_ny_flags(ts: datetime | pd.Timestamp) -> tuple[bool, bool]:
    t = pd.Timestamp(ts)
    if t.tz is None:
        t = t.tz_localize("UTC")
    lon = t.tz_convert("Europe/London")
    ny = t.tz_convert("America/New_York")
    lt = lon.time()
    nt = ny.time()
    london_ok = time(8, 0) <= lt < time(13, 0)
    ny_ok = time(9, 30) <= nt < time(14, 0)
    return london_ok, ny_ok


def session_label_for_sweep_bar(ts: pd.Timestamp | None) -> str:
    """Human-readable session bucket for the bar that printed the sweep."""
    if ts is None:
        return "n/a"
    l_ok, n_ok = _london_ny_flags(ts)
    if l_ok and n_ok:
        return "London + New York (hours overlap on this bar)"
    if l_ok:
        return "London session window"
    if n_ok:
        return "New York session window"
    return "Outside labeled windows (unexpected for sweep bar)"


def _seg_after(
    df_ohlc: pd.DataFrame,
    sweep_ts: pd.Timestamp,
    cfg: LiquidityMSSConfig,
) -> pd.DataFrame:
    cols = ["open", "high", "low", "close"]
    end_scan = sweep_ts + pd.Timedelta(hours=cfg.mss_forward_hours)
    return df_ohlc.loc[(df_ohlc.index > sweep_ts) & (df_ohlc.index <= end_scan), cols].copy()


def explain_accepted(
    sweep_side: str,
    mss_direction: str,
    sweep_ts: pd.Timestamp | None,
    ah: float,
    al: float,
    cfg: LiquidityMSSConfig,
) -> str:
    parts = []
    if sweep_side == "above":
        parts.append(
            f"The first qualifying sweep took liquidity above the Asian high ({ah:.5f}) "
            f"during the post-Asian London/NY filter; wick/body crossed that level."
        )
        parts.append(
            "A bearish displacement candle appeared within the forward MSS window "
            f"(next {cfg.mss_forward_hours}h after the sweep bar): strong red body vs range "
            f"(≥{cfg.displacement_body_ratio:.0%} body share, range vs median thresholds)."
        )
        parts.append(
            "MSS bearish was accepted because price later closed below the fractal swing-low "
            "anchor after that displacement (or the fallback running-low break when fractals were sparse)."
        )
    else:
        parts.append(
            f"The first qualifying sweep took liquidity below the Asian low ({al:.5f}) "
            "under the same session filters."
        )
        parts.append(
            "A bullish displacement candle met the impulsive criteria within the forward window."
        )
        parts.append(
            "MSS bullish was accepted because price later closed above the fractal swing-high "
            "anchor after displacement (or the fallback break higher when fractals were sparse)."
        )
    parts.append(
        f"Sweep bar timestamp (UTC): {sweep_ts} — session label reflects local London/NY kill-zones."
    )
    return " ".join(parts)


def explain_rejection(
    sweep_side: str,
    seg_after: pd.DataFrame,
    cfg: LiquidityMSSConfig,
    sweep_ts: pd.Timestamp,
) -> str:
    if seg_after.empty:
        return (
            "MSS could not be evaluated: no subsequent bars exist within the configured "
            f"{cfg.mss_forward_hours}h forward window after the sweep (data ends or gap)."
        )

    ref_range = _segment_range_ref(seg_after)

    if sweep_side == "above":
        ok_mss = _bearish_mss_after_sweep_up(seg_after, cfg, sweep_ts)
        disp_idx = None
        for ts, row in seg_after.iterrows():
            if ts <= sweep_ts:
                continue
            if _is_bearish_displacement(
                row,
                ref_range,
                cfg.displacement_body_ratio,
                cfg.displacement_min_range_ratio,
            ):
                disp_idx = ts
                break
        if disp_idx is None:
            return (
                "Rejected — no bearish displacement candle after the sweep within the forward "
                f"window. Rules require a red candle whose body is at least "
                f"{cfg.displacement_body_ratio:.0%} of its range and whose range is not tiny vs "
                f"the median range in that window (≥{cfg.displacement_min_range_ratio:.0%} of median). "
                "Without displacement, MSS is not allowed to validate."
            )
        if not ok_mss:
            return (
                "Rejected — bearish displacement did occur, but MSS stayed unconfirmed: "
                "no qualifying close below the swing-low anchor built from post-displacement fractals "
                "(or the simplified fallback close below running low never triggered). "
                "Structure therefore did not register the bearish shift under the coded fractal rules."
            )
        return (
            "Audit note: this day was tagged rejected but bearish MSS logic returned true — "
            "re-check data vs daily table (should be rare)."
        )
    else:
        ok_mss = _bullish_mss_after_sweep_down(seg_after, cfg, sweep_ts)
        disp_idx = None
        for ts, row in seg_after.iterrows():
            if ts <= sweep_ts:
                continue
            if _is_bullish_displacement(
                row,
                ref_range,
                cfg.displacement_body_ratio,
                cfg.displacement_min_range_ratio,
            ):
                disp_idx = ts
                break
        if disp_idx is None:
            return (
                "Rejected — no bullish displacement after the sweep within the forward window "
                f"(same body/range thresholds as configured). MSS requires that impulse first."
            )
        if not ok_mss:
            return (
                "Rejected — bullish displacement occurred, but no confirmed close above the "
                "post-displacement fractal swing-high anchor (and fallback break higher did not trigger). "
                "The bullish MSS leg failed despite the liquidity sweep."
            )
        return (
            "Audit note: tagged rejected but bullish MSS logic returned true — rare inconsistency."
        )

    return "Rejected — reason could not be classified (unexpected)."


def _resolve_sweep_bar(
    df_ann: pd.DataFrame,
    killzone: pd.Series,
    D,
    ah: float,
    al: float,
) -> tuple[str, pd.Timestamp | None]:
    sm = sweep_session_mask(df_ann, D, killzone)
    seg = df_ann.loc[sm, ["open", "high", "low", "close"]].copy()
    return _first_liquidity_sweep(seg, ah, al)


def _stratified_pick(rows: list[dict], n: int, key_order: list[str]) -> list[dict]:
    """Round-robin newest-first within each asset to spread examples."""
    by_asset = {a: [] for a in key_order}
    for r in sorted(rows, key=lambda x: x["session_date"], reverse=True):
        by_asset[r["asset"]].append(r)
    out: list[dict] = []
    i = 0
    while len(out) < n and any(by_asset[a] for a in key_order):
        a = key_order[i % len(key_order)]
        if by_asset[a]:
            out.append(by_asset[a].pop(0))
        i += 1
        if i > n * len(key_order) + 50:
            break
    return out


def run_audit(
    *,
    n_accepted: int = 30,
    n_rejected: int = 10,
) -> None:
    cfg = default_liquidity_mss_config()

    # Collect candidate + rejected rows with asset tags
    candidates: list[dict] = []
    rejected_sweep: list[dict] = []

    for sym in sorted(SUPPORTED_OHLC_SYMBOLS):
        ohlcv = load_symbol_ohlcv_csv(sym)
        daily = detect_asian_liquidity_mss(ohlcv, cfg, symbol=sym, eurusd_institutional=True)
        df_ann = annotate_tokyo_calendar(ohlcv)
        killzone = build_london_ny_killzone_mask(df_ann.index)

        for _, row in daily.iterrows():
            rec = {
                "asset": sym,
                "session_date": row["session_date"],
                "asian_high": float(row["asian_high"]),
                "asian_low": float(row["asian_low"]),
                "sweep_side": row["sweep_side"],
                "mss_direction": row["mss_direction"],
                "tag": row["tag"],
                "_df_ann": df_ann,
                "_killzone": killzone,
                "_ohlc": ohlcv,
            }
            if "candidate" in str(row["tag"]):
                candidates.append(rec)
            elif row["sweep_side"] != "none" and row["tag"] == "no setup":
                rejected_sweep.append(rec)

    order = ["EURUSD", "GBPUSD", "XAUUSD", "NAS100"]
    picked_acc = _stratified_pick(candidates, n_accepted, order)
    picked_rej = _stratified_pick(rejected_sweep, n_rejected, order)

    print("=" * 72)
    print("ASIAN LIQUIDITY SWEEP + MSS — AUDIT REPORT (READ-ONLY)")
    print("=" * 72)
    print()
    print(
        f"Config snapshot: Asian window from project defaults; MSS forward window = "
        f"{cfg.mss_forward_hours}h; displacement body ≥ {cfg.displacement_body_ratio:.0%}; "
        f"range vs median ≥ {cfg.displacement_min_range_ratio:.0%}; fractals {cfg.fractal_left}+{cfg.fractal_right}."
    )
    print()

    # --- Accepted ---
    print("-" * 72)
    print(f"SECTION A — {len(picked_acc)} ACCEPTED SETUP EXAMPLES (stratified across assets)")
    print("-" * 72)
    for i, rec in enumerate(picked_acc, 1):
        df_ann = rec["_df_ann"]
        killzone = rec["_killzone"]
        ah, al = rec["asian_high"], rec["asian_low"]
        D = rec["session_date"]
        _ss, sweep_ts = _resolve_sweep_bar(df_ann, killzone, D, ah, al)
        sess = session_label_for_sweep_bar(sweep_ts)

        print(f"\n### Example {i} — ACCEPTED ({rec['asset']})")
        print(f"  Date (Asian session_date / Tokyo): {rec['session_date']}")
        print(f"  Asset:                             {rec['asset']}")
        print(f"  Asian High:                        {ah:.5f}")
        print(f"  Asian Low:                         {al:.5f}")
        print(f"  Sweep side:                        {rec['sweep_side']} Asian range")
        print(f"  MSS direction:                     {rec['mss_direction']}")
        print(f"  Sweep occurred in:                 {sess}")
        print()
        print("  Why accepted:")
        print("   ", explain_accepted(rec["sweep_side"], rec["mss_direction"], sweep_ts, ah, al, cfg))

    # --- Rejected ---
    print()
    print("-" * 72)
    print(f"SECTION B — {len(picked_rej)} BORDERLINE CASES (SWEEP YES, MSS NO)")
    print("-" * 72)
    for i, rec in enumerate(picked_rej, 1):
        df_ann = rec["_df_ann"]
        killzone = rec["_killzone"]
        ah, al = rec["asian_high"], rec["asian_low"]
        D = rec["session_date"]
        ss, sweep_ts = _resolve_sweep_bar(df_ann, killzone, D, ah, al)
        seg_after = (
            _seg_after(rec["_ohlc"], sweep_ts, cfg)
            if sweep_ts is not None
            else pd.DataFrame()
        )
        sess = session_label_for_sweep_bar(sweep_ts)

        print(f"\n### Example {i} — REJECTED ({rec['asset']})")
        print(f"  Date (Asian session_date / Tokyo): {rec['session_date']}")
        print(f"  Asset:                             {rec['asset']}")
        print(f"  Asian High:                        {ah:.5f}")
        print(f"  Asian Low:                         {al:.5f}")
        print(f"  Sweep side:                        {rec['sweep_side']} Asian range")
        print(f"  MSS direction (result):            none / not validated")
        print(f"  Sweep occurred in:                 {sess}")
        print()
        print("  Why rejected:")
        print(
            "   ",
            explain_rejection(rec["sweep_side"], seg_after, cfg, sweep_ts)
            if sweep_ts is not None
            else "   Sweep timestamp missing (unexpected).",
        )

    print()
    print("=" * 72)
    print("END OF AUDIT")
    print("=" * 72)


def main() -> int:
    run_audit()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

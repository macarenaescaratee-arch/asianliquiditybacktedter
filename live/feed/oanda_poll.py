"""Poll OANDA REST for completed H1 mid candles (EUR_USD, etc.)."""

from __future__ import annotations

import pandas as pd

from live.broker.oanda_rest import OandaRestClient
from live.live_types import Candle, candle_to_timestamp


def _mid_bar_to_candle(row: dict) -> Candle | None:
    if not row.get("complete"):
        return None
    mid = row.get("mid") or {}
    if not all(k in mid for k in ("o", "h", "l", "c")):
        return None
    ts_raw = row.get("time")
    if not ts_raw:
        return None
    ts = pd.Timestamp(ts_raw)
    vol = row.get("volume")
    v = float(vol) if vol not in (None, "") else None
    return Candle(
        ts=candle_to_timestamp(ts),
        open=float(mid["o"]),
        high=float(mid["h"]),
        low=float(mid["l"]),
        close=float(mid["c"]),
        volume=v,
    )


def bootstrap_closed_candles(
    client: OandaRestClient,
    instrument: str,
    *,
    count: int,
) -> list[Candle]:
    raw = client.get_candles(instrument, count=min(max(count, 1), 5000))
    out: list[Candle] = []
    for row in raw:
        c = _mid_bar_to_candle(row)
        if c is not None:
            out.append(c)
    out.sort(key=lambda x: x.ts)
    return out


def poll_new_closed_candles(
    client: OandaRestClient,
    instrument: str,
    *,
    since_ts: pd.Timestamp | None,
    count: int = 25,
) -> list[Candle]:
    raw = client.get_candles(instrument, count=min(max(count, 2), 500))
    out: list[Candle] = []
    for row in raw:
        c = _mid_bar_to_candle(row)
        if c is None:
            continue
        if since_ts is not None and c.ts <= since_ts:
            continue
        out.append(c)
    out.sort(key=lambda x: x.ts)
    dedup: dict[pd.Timestamp, Candle] = {c.ts: c for c in out}
    return [dedup[k] for k in sorted(dedup.keys())]

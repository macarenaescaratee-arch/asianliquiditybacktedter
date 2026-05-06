"""
Backtest engine placeholder.

Future use: orchestrate data → strategy signals → vectorbt.Portfolio (or custom PnL),
apply fees/spread from ``config``, and return structured results for ``analytics`` and
``reports``. This module stays thin; heavy lifting belongs in vectorbt or dedicated runners.
"""

from __future__ import annotations

from typing import Any

import pandas as pd


def run_stub_backtest(ohlcv: pd.DataFrame) -> dict[str, Any]:
    """
    Non-functional stub documenting the future return shape.

    Returns summary dict placeholders until real portfolio construction exists.
    """
    return {
        "status": "stub",
        "rows": len(ohlcv),
        "message": "Implement vectorbt Portfolio in a later phase.",
    }

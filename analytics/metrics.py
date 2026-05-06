"""
Performance and risk metrics.

Future use: functions that take equity curves or vectorbt results and return standardized
metric tables; optional matplotlib/plotly helpers for quick QA plots outside Streamlit.
"""

from __future__ import annotations

from typing import Any


def summarize_stub(portfolio_result: Any) -> dict[str, Any]:
    """Placeholder until real metrics wrap vectorbt stats or numpy/pandas series."""
    return {"status": "stub", "input_type": type(portfolio_result).__name__}

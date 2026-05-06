"""
Map project symbols (EURUSD, …) to ``dukascopy_python.instruments`` identifiers.

Dukascopy lists the US Tech 100 cash CFD as ``INSTRUMENT_US_TECH_US_USD``; we treat it
as the research proxy for **NAS100** in this repository.
"""

from __future__ import annotations

import dukascopy_python.instruments as dki

from data.symbols import validate_symbol

_INSTRUMENT_BY_SYMBOL: dict[str, str] = {
    "EURUSD": "INSTRUMENT_FX_MAJORS_EUR_USD",
    "GBPUSD": "INSTRUMENT_FX_MAJORS_GBP_USD",
    "XAUUSD": "INSTRUMENT_FX_METALS_XAU_USD",
    # Nasdaq-100 proxy on Dukascopy (US Tech 100 index CFD, USD margin)
    "NAS100": "INSTRUMENT_US_TECH_US_USD",
}


def dukascopy_instrument_for(symbol: str):
    """Return the ``dukascopy_python.instruments`` object for a supported symbol."""
    key = validate_symbol(symbol)
    name = _INSTRUMENT_BY_SYMBOL[key]
    return getattr(dki, name)

"""
Cross-cutting helper functions.

Future use: safe datetime parsing, session window masks (e.g. Tokyo equity auction),
path helpers beyond ``config``, and consistent random seeds for reproducible research.
"""

from __future__ import annotations

import logging
from typing import Final

LOGGER_NAME: Final[str] = "asian_liquidity_backtester"


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a module-level logger; configure handlers in main/dashboard entrypoints."""
    return logging.getLogger(name or LOGGER_NAME)

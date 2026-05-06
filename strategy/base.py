"""
Abstract strategy scaffolding.

Future use: a small Strategy protocol or base class defining inputs (prices, features)
and outputs (entries, exits, sizing). Concrete strategies will subclass or implement
this contract for consistent integration with vectorbt and the custom engine.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import pandas as pd


class StrategyBase(ABC):
    """Placeholder interface for strategy implementations."""

    name: str = "unnamed"

    @abstractmethod
    def generate_signals(self, ohlcv: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
        """
        Return a DataFrame of signals aligned with ``ohlcv`` index.

        Convention (to be finalized): columns such as ``entries``, ``exits``, or
        position targets consumable by vectorbt's ``from_signals``.
        """
        raise NotImplementedError

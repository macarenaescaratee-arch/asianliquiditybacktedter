"""Broker abstraction — wire to MT5, Oanda, IB, etc."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from live.account_types import AccountSnapshot, FillEvent
from live.execution_plan import LiveExecutionPlan


@dataclass
class BrokerOrderResult:
    client_order_id: str
    accepted: bool
    message: str
    raw: dict[str, Any] | None = None
    broker_trade_id: str | None = None
    broker_order_ids: list[str] | None = None


class BrokerClient(ABC):
    @abstractmethod
    def place_market_entry_with_bracket(
        self,
        plan: LiveExecutionPlan,
        *,
        units: float,
        client_order_id: str,
    ) -> BrokerOrderResult:
        """Market (or aggressive limit) entry with attached SL/TP in native units."""
        raise NotImplementedError

    @abstractmethod
    def cancel_all(self, symbol: str) -> None:
        raise NotImplementedError

    def suggest_units(self, plan: LiveExecutionPlan) -> float:
        """Position size in broker-native units (e.g. OANDA base units)."""
        return 1.0

    def cancel_order(self, order_id: str) -> None:
        """Cancel a single pending order by venue order id."""
        del order_id

    def modify_trade_brackets(
        self,
        trade_id: str,
        *,
        stop_loss: float | None = None,
        take_profit: float | None = None,
    ) -> None:
        """Update SL/TP on an open trade when the venue models brackets as trade attachments."""
        del trade_id, stop_loss, take_profit

    def sync_account(self) -> AccountSnapshot | None:
        """Latest balances / margin for risk checks (optional)."""
        return None

    def poll_fills(self) -> list[FillEvent]:
        """New fill events since the last poll cursor (broker-managed)."""
        return []

    def close_open_trade(self, trade_id: str | None) -> None:
        """Flatten a venue position when the model exits (best-effort)."""
        del trade_id

    def export_runtime_state(self) -> dict:
        """Broker-specific resume state (e.g., transaction cursor)."""
        return {}

    def import_runtime_state(self, payload: dict) -> None:
        """Restore broker-specific resume state."""
        del payload

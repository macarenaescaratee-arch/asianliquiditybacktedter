"""Paper broker: logs intended orders, no network."""

from __future__ import annotations

import uuid
from typing import Any

from live.account_types import AccountSnapshot, FillEvent
from live.broker.base import BrokerClient, BrokerOrderResult
from live.execution_plan import LiveExecutionPlan


class PaperBrokerClient(BrokerClient):
    def __init__(self, *, default_units: float = 1.0) -> None:
        self._default_units = float(default_units)
        self.intents: list[dict[str, Any]] = []

    def suggest_units(self, plan: LiveExecutionPlan) -> float:
        del plan
        return self._default_units

    def place_market_entry_with_bracket(
        self,
        plan: LiveExecutionPlan,
        *,
        units: float,
        client_order_id: str,
    ) -> BrokerOrderResult:
        rec = {
            "id": client_order_id or str(uuid.uuid4()),
            "symbol": plan.symbol,
            "direction": plan.direction,
            "units": units,
            "entry": plan.entry_price,
            "sl": plan.stop_price,
            "tp": plan.take_profit_price,
            "risk_pips": plan.risk_pips,
        }
        self.intents.append(rec)
        return BrokerOrderResult(
            client_order_id=rec["id"],
            accepted=True,
            message="paper_accepted",
            raw=rec,
            broker_trade_id=f"paper_{rec['id']}",
        )

    def cancel_all(self, symbol: str) -> None:
        self.intents.append({"action": "cancel_all", "symbol": symbol})

    def cancel_order(self, order_id: str) -> None:
        self.intents.append({"action": "cancel_order", "order_id": order_id})

    def modify_trade_brackets(
        self,
        trade_id: str,
        *,
        stop_loss: float | None = None,
        take_profit: float | None = None,
    ) -> None:
        self.intents.append(
            {
                "action": "modify_trade_brackets",
                "trade_id": trade_id,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
            }
        )

    def sync_account(self) -> AccountSnapshot | None:
        return AccountSnapshot(
            balance=100_000.0,
            currency="USD",
            margin_available=100_000.0,
            nav=100_000.0,
        )

    def poll_fills(self) -> list[FillEvent]:
        return []

    def close_open_trade(self, trade_id: str | None) -> None:
        self.intents.append({"action": "close_open_trade", "trade_id": trade_id})

    def export_runtime_state(self) -> dict:
        return {}

    def import_runtime_state(self, payload: dict) -> None:
        del payload

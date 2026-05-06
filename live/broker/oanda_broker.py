"""OANDA v20 execution bridge for the live EURUSD bot."""

from __future__ import annotations

import logging
import math
from typing import Any

from live.account_types import AccountSnapshot, FillEvent, RiskBudget
from live.broker.base import BrokerClient, BrokerOrderResult
from live.broker.oanda_rest import (
    OandaRestClient,
    OandaRestError,
    format_price_eurusd,
    parse_account_snapshot,
    parse_fill_events,
)
from live.execution_plan import LiveExecutionPlan

LOG = logging.getLogger(__name__)


def _venue_symbol(plan_symbol: str) -> str:
    s = plan_symbol.upper().replace("/", "")
    if s == "EURUSD":
        return "EUR_USD"
    if "_" in s:
        return s
    if len(s) == 6:
        return f"{s[:3]}_{s[3:]}"
    return s


def _extract_trade_id_from_order_response(data: dict[str, Any]) -> str | None:
    oft = data.get("orderFillTransaction")
    if not isinstance(oft, dict):
        return None
    to = oft.get("tradeOpened")
    if isinstance(to, dict) and to.get("tradeID"):
        return str(to.get("tradeID"))
    opened = oft.get("tradesOpened")
    if isinstance(opened, list) and opened:
        t0 = opened[0]
        if isinstance(t0, dict) and t0.get("tradeID"):
            return str(t0.get("tradeID"))
    return None


class OandaExecutionBroker(BrokerClient):
    """
    Routes ``LiveExecutionPlan`` to OANDA market orders with on-fill SL/TP.

    Requires a configured ``OandaRestClient`` (practice or live).
    """

    def __init__(
        self,
        client: OandaRestClient,
        *,
        instrument: str = "EUR_USD",
        risk_fraction: float = 0.005,
        dollars_per_pip_per_10k: float = 1.0,
        fallback_units: float = 10_000.0,
    ) -> None:
        self._client = client
        self._instrument = instrument
        self._risk_fraction = float(risk_fraction)
        self._dpp = float(dollars_per_pip_per_10k)
        self._fallback = float(fallback_units)
        self._txn_cursor: str = ""
        self._bootstrap_txn_cursor()

    def _bootstrap_txn_cursor(self) -> None:
        try:
            acc = self._client.get_account().get("account") or {}
            self._txn_cursor = str(acc.get("lastTransactionID") or "")
        except OandaRestError as exc:
            LOG.warning("Could not seed transaction cursor: %s", exc)
            self._txn_cursor = ""

    def suggest_units(self, plan: LiveExecutionPlan) -> float:
        try:
            summ = self._client.get_account_summary()
            snap = parse_account_snapshot(summ)
            budget = RiskBudget(snap, self._risk_fraction)
            u = budget.suggested_units_simple(
                plan.risk_pips,
                dollars_per_pip_per_10k=self._dpp,
            )
            if u > 0:
                return float(u)
        except (OandaRestError, TypeError, ValueError) as exc:
            LOG.warning("suggest_units using fallback: %s", exc)
        return self._fallback

    def place_market_entry_with_bracket(
        self,
        plan: LiveExecutionPlan,
        *,
        units: float,
        client_order_id: str,
    ) -> BrokerOrderResult:
        inst = _venue_symbol(plan.symbol)
        base = abs(float(units)) if float(units) != 0 else abs(self.suggest_units(plan))
        signed = base if plan.direction == "long" else -base
        sl = format_price_eurusd(plan.stop_price)
        tp: str | None
        if plan.take_profit_price is not None and not math.isnan(plan.take_profit_price):
            tp = format_price_eurusd(float(plan.take_profit_price))
        else:
            tp = None
        try:
            data = self._client.place_market_order(
                inst,
                units=signed,
                stop_loss=sl,
                take_profit=tp,
            )
        except OandaRestError as exc:
            return BrokerOrderResult(
                client_order_id=client_order_id,
                accepted=False,
                message=str(exc),
                raw=None,
            )
        tid = _extract_trade_id_from_order_response(data)
        return BrokerOrderResult(
            client_order_id=client_order_id,
            accepted=True,
            message="oanda_filled",
            raw=data,
            broker_trade_id=tid,
        )

    def cancel_all(self, symbol: str) -> None:
        inst = _venue_symbol(symbol)
        for t in self._client.get_open_trades():
            if str(t.get("instrument")) == inst:
                try:
                    self._client.close_trade(str(t.get("id")))
                except OandaRestError as exc:
                    LOG.warning("close trade %s: %s", t.get("id"), exc)
        for o in self._client.get_pending_orders():
            if str(o.get("instrument")) == inst:
                try:
                    self._client.cancel_order(str(o.get("id")))
                except OandaRestError as exc:
                    LOG.warning("cancel order %s: %s", o.get("id"), exc)

    def cancel_order(self, order_id: str) -> None:
        self._client.cancel_order(order_id)

    def modify_trade_brackets(
        self,
        trade_id: str,
        *,
        stop_loss: float | None = None,
        take_profit: float | None = None,
    ) -> None:
        sl_s = format_price_eurusd(stop_loss) if stop_loss is not None else None
        tp_s = format_price_eurusd(take_profit) if take_profit is not None else None
        if sl_s is None and tp_s is None:
            return
        self._client.replace_trade_orders(trade_id, stop_loss=sl_s, take_profit=tp_s)

    def sync_account(self) -> AccountSnapshot | None:
        try:
            return parse_account_snapshot(self._client.get_account_summary())
        except OandaRestError as exc:
            LOG.warning("sync_account: %s", exc)
            return None

    def poll_fills(self) -> list[FillEvent]:
        if not self._txn_cursor:
            self._bootstrap_txn_cursor()
        if not self._txn_cursor:
            return []
        try:
            txns, last = self._client.transactions_since(self._txn_cursor)
        except OandaRestError as exc:
            LOG.warning("poll_fills: %s", exc)
            return []
        fills = parse_fill_events(txns)
        if last:
            self._txn_cursor = last
        elif txns:
            self._txn_cursor = str(txns[-1].get("id", self._txn_cursor))
        return fills

    def close_open_trade(self, trade_id: str | None) -> None:
        if not trade_id or str(trade_id).startswith("paper_"):
            return
        try:
            self._client.close_trade(str(trade_id))
        except OandaRestError as exc:
            LOG.warning("close_open_trade %s: %s", trade_id, exc)

    def export_runtime_state(self) -> dict:
        return {"txn_cursor": self._txn_cursor}

    def import_runtime_state(self, payload: dict) -> None:
        c = str((payload or {}).get("txn_cursor") or "").strip()
        if c:
            self._txn_cursor = c

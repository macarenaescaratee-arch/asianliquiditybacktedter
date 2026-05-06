"""
OANDA v20 REST client: candles, account, orders, trades, transactions.

Requires: ``requests``. Env: ``OANDA_API_TOKEN``, ``OANDA_ACCOUNT_ID``, ``OANDA_ENV`` (practice|live).

Optional: ``OANDA_REST_BASE`` — override API root (e.g. local mock ``http://127.0.0.1:18999``).
"""

from __future__ import annotations

import logging
import os
from typing import Any

import requests

from live.account_types import AccountSnapshot, FillEvent, PositionSnapshot

LOG = logging.getLogger(__name__)

PRACTICE_HOST = "https://api-fxpractice.oanda.com"
LIVE_HOST = "https://api-fxtrade.oanda.com"


class OandaRestError(RuntimeError):
    pass


class OandaRestClient:
    def __init__(
        self,
        *,
        api_token: str,
        account_id: str,
        practice: bool = True,
        timeout: float = 30.0,
    ) -> None:
        self._token = api_token
        self.account_id = account_id
        override = os.environ.get("OANDA_REST_BASE", "").strip().rstrip("/")
        if override:
            self._base = override
        elif practice:
            self._base = PRACTICE_HOST
        else:
            self._base = LIVE_HOST
        self._timeout = timeout
        self._session = requests.Session()
        self._session.headers.update(
            {
                "Authorization": f"Bearer {api_token}",
                "Content-Type": "application/json",
            }
        )

    def _url(self, path: str) -> str:
        p = path if path.startswith("/") else f"/{path}"
        return f"{self._base}{p}"

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json_body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        url = self._url(path)
        r = self._session.request(
            method,
            url,
            params=params,
            json=json_body,
            timeout=self._timeout,
        )
        try:
            data = r.json()
        except Exception:
            data = {}
        if r.status_code >= 400:
            msg = data.get("errorMessage") or data.get("message") or r.text
            raise OandaRestError(f"{r.status_code} {method} {path}: {msg}")
        return data if isinstance(data, dict) else {}

    def get_candles(
        self,
        instrument: str,
        *,
        granularity: str = "H1",
        count: int = 500,
        price: str = "M",
    ) -> list[dict[str, Any]]:
        path = f"/v3/instruments/{instrument}/candles"
        data = self._request(
            "GET",
            path,
            params={
                "granularity": granularity,
                "price": price,
                "count": min(count, 5000),
            },
        )
        return list(data.get("candles") or [])

    def get_account_summary(self) -> dict[str, Any]:
        path = f"/v3/accounts/{self.account_id}/summary"
        return self._request("GET", path)

    def get_account(self) -> dict[str, Any]:
        path = f"/v3/accounts/{self.account_id}"
        return self._request("GET", path)

    def get_open_trades(self) -> list[dict[str, Any]]:
        path = f"/v3/accounts/{self.account_id}/openTrades"
        data = self._request("GET", path)
        return list(data.get("trades") or [])

    def get_pending_orders(self) -> list[dict[str, Any]]:
        path = f"/v3/accounts/{self.account_id}/pendingOrders"
        data = self._request("GET", path)
        return list(data.get("orders") or [])

    def place_market_order(
        self,
        instrument: str,
        *,
        units: float,
        stop_loss: str | None = None,
        take_profit: str | None = None,
        client_tag: str | None = None,
    ) -> dict[str, Any]:
        order: dict[str, Any] = {
            "type": "MARKET",
            "instrument": instrument,
            "units": str(int(units)) if units == int(units) else str(units),
            "timeInForce": "FOK",
            "positionFill": "DEFAULT",
        }
        if stop_loss:
            order["stopLossOnFill"] = {"price": stop_loss}
        if take_profit:
            order["takeProfitOnFill"] = {"price": take_profit}
        if client_tag:
            order["clientExtensions"] = {"tag": client_tag[:128]}
        body = {"order": order}
        path = f"/v3/accounts/{self.account_id}/orders"
        return self._request("POST", path, json_body=body)

    def cancel_order(self, order_id: str) -> dict[str, Any]:
        path = f"/v3/accounts/{self.account_id}/orders/{order_id}/cancel"
        return self._request("PUT", path)

    def replace_trade_orders(
        self,
        trade_id: str,
        *,
        stop_loss: str | None = None,
        take_profit: str | None = None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {}
        if stop_loss is not None:
            body["stopLoss"] = {"price": stop_loss, "timeInForce": "GTC"}
        if take_profit is not None:
            body["takeProfit"] = {"price": take_profit, "timeInForce": "GTC"}
        path = f"/v3/accounts/{self.account_id}/trades/{trade_id}/orders"
        return self._request("PUT", path, json_body=body)

    def close_trade(self, trade_id: str, *, units: str | None = None) -> dict[str, Any]:
        body: dict[str, Any] = {}
        if units:
            body["units"] = units
        path = f"/v3/accounts/{self.account_id}/trades/{trade_id}/close"
        return self._request("PUT", path, json_body=body)

    def transactions_since(self, from_id: str) -> tuple[list[dict[str, Any]], str | None]:
        path = f"/v3/accounts/{self.account_id}/transactions/sinceid"
        data = self._request("GET", path, params={"id": from_id})
        txns = list(data.get("transactions") or [])
        last = data.get("lastTransactionID")
        return txns, str(last) if last is not None else None


def parse_account_snapshot(payload: dict[str, Any]) -> AccountSnapshot:
    acc = payload.get("account") or payload
    bal = float(acc.get("balance", 0) or 0)
    cur = str(acc.get("currency") or "USD")
    margin_avail = float(acc.get("marginAvailable", bal) or bal)
    margin_used = float(acc.get("marginUsed", 0) or 0)
    nav = float(acc.get("NAV", bal) or bal)
    upl = float(acc.get("unrealizedPL", 0) or 0)
    return AccountSnapshot(
        balance=bal,
        currency=cur,
        margin_available=margin_avail,
        nav=nav,
        margin_used=margin_used,
        unrealized_pl=upl,
        raw=acc,
    )


def parse_open_positions(trades: list[dict[str, Any]]) -> list[PositionSnapshot]:
    out: list[PositionSnapshot] = []
    for t in trades:
        tid = str(t.get("id", ""))
        inst = str(t.get("instrument", ""))
        u = float(t.get("currentUnits", 0) or 0)
        direction = "long" if u > 0 else "short"
        avg = float(t.get("price", 0) or 0)
        upl = float(t.get("unrealizedPL", 0) or 0)
        sl = t.get("stopLossOrder")
        tp = t.get("takeProfitOrder")
        sl_px = float(sl.get("price")) if isinstance(sl, dict) and sl.get("price") else None
        tp_px = float(tp.get("price")) if isinstance(tp, dict) and tp.get("price") else None
        out.append(
            PositionSnapshot(
                trade_id=tid,
                instrument=inst,
                direction=direction,
                units=abs(u),
                avg_price=avg,
                unrealized_pl=upl,
                stop_loss=sl_px,
                take_profit=tp_px,
                raw=t,
            )
        )
    return out


def _txn_to_fills(txn: dict[str, Any]) -> list[FillEvent]:
    tid = str(txn.get("id", ""))
    ttype = str(txn.get("type", ""))
    out: list[FillEvent] = []
    if ttype == "ORDER_FILL":
        inst = str(txn.get("instrument", ""))
        t_utc = str(txn.get("time", ""))
        u = float(txn.get("units", 0) or 0)
        px = float(txn.get("price", 0) or 0)
        pl = float(txn.get("pl", 0) or 0)
        fin = float(txn.get("financing", 0) or 0)
        out.append(
            FillEvent(
                transaction_id=tid,
                instrument=inst,
                time_utc=t_utc,
                units=u,
                price=px,
                pl=pl,
                financing=fin,
                raw=txn,
            )
        )
    return out


def parse_fill_events(transactions: list[dict[str, Any]]) -> list[FillEvent]:
    fills: list[FillEvent] = []
    for tx in transactions:
        fills.extend(_txn_to_fills(tx))
    return fills


def format_price_eurusd(x: float) -> str:
    return f"{x:.5f}"

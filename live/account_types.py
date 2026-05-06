"""Account, position, and fill snapshots for risk sync and reconciliation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class AccountSnapshot:
    balance: float
    currency: str
    margin_available: float
    nav: float
    margin_used: float = 0.0
    unrealized_pl: float = 0.0
    raw: dict | None = None


@dataclass(slots=True)
class PositionSnapshot:
    trade_id: str
    instrument: str
    direction: str
    units: float
    avg_price: float
    unrealized_pl: float
    stop_loss: float | None = None
    take_profit: float | None = None
    raw: dict | None = None


@dataclass(slots=True)
class FillEvent:
    transaction_id: str
    instrument: str
    time_utc: str
    units: float
    price: float
    pl: float
    financing: float = 0.0
    raw: dict | None = None


@dataclass(slots=True)
class RiskBudget:
    """Sizing hint from account sync (caller converts to broker units)."""

    account: AccountSnapshot
    risk_fraction_of_balance: float
    max_units_cap: float | None = None

    def notional_at_risk_quote(
        self,
        risk_pips: float,
        *,
        quote_per_pip_per_unit: float,
    ) -> float:
        """Approximate quote currency at risk for 1 unit over ``risk_pips`` (broker-specific)."""
        return abs(risk_pips) * quote_per_pip_per_unit

    def suggested_units_simple(
        self,
        risk_pips: float,
        *,
        dollars_per_pip_per_10k: float = 1.0,
    ) -> float:
        """
        Crude EURUSD demo sizing: balance * risk_fraction / (risk_pips * $/pip per 10k units).

        ``dollars_per_pip_per_10k`` ~ 1.0 for USD-denominated account on EURUSD mini semantics.
        Override with measured pip value from your broker.
        """
        if risk_pips <= 0:
            return 0.0
        risk_cash = self.account.balance * self.risk_fraction_of_balance
        denom = risk_pips * dollars_per_pip_per_10k / 10000.0
        if denom <= 0:
            return 0.0
        u = risk_cash / denom
        if self.max_units_cap is not None:
            u = min(u, self.max_units_cap)
        return max(0.0, float(u))

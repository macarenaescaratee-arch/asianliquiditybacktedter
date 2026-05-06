"""Open-position lifecycle: Phase 4 live stepping + optional broker routing."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date

import pandas as pd

from backtester.asian_mss_execution import TradeSetup
from backtester.eurusd_phase4_execution import (
    ExecutionVariant,
    Phase4LiveState,
    compute_atr,
    phase4_live_init,
    phase4_live_step,
)
from live.broker.base import BrokerClient
from live.execution_plan import LiveExecutionPlan
from live.trade_logger import TradeLogger


@dataclass
class ManagedPosition:
    setup: TradeSetup
    plan: LiveExecutionPlan
    state: Phase4LiveState
    client_order_id: str
    broker_trade_id: str | None = None


class QuantTradeManager:
    """
    One position max. Uses ``phase4_live_step`` for parity with backtest Phase 4 rules.
    """

    def __init__(
        self,
        execution: ExecutionVariant,
        broker: BrokerClient,
        logger: TradeLogger,
    ) -> None:
        self.execution = execution
        self.broker = broker
        self.logger = logger
        self.position: ManagedPosition | None = None
        self.metrics: dict[str, int] = {
            "detected_setups": 0,
            "submitted_intents": 0,
            "open_events": 0,
            "close_events": 0,
        }

    def snapshot(self) -> dict:
        if self.position is None:
            return {"active": False, "metrics": dict(self.metrics)}
        p = self.position
        return {
            "active": True,
            "setup": _setup_to_dict(p.setup),
            "plan": _plan_to_dict(p.plan),
            "state": _state_to_dict(p.state),
            "client_order_id": p.client_order_id,
            "broker_trade_id": p.broker_trade_id,
            "metrics": dict(self.metrics),
        }

    def restore_from_snapshot(self, payload: dict) -> bool:
        if not payload or not payload.get("active"):
            m = payload.get("metrics") if isinstance(payload, dict) else None
            if isinstance(m, dict):
                for k in self.metrics:
                    if k in m:
                        self.metrics[k] = int(m[k])
            self.position = None
            return True
        try:
            setup = _setup_from_dict(payload["setup"])
            plan = _plan_from_dict(payload["plan"])
            state = _state_from_dict(payload["state"])
            self.position = ManagedPosition(
                setup=setup,
                plan=plan,
                state=state,
                client_order_id=str(payload.get("client_order_id", "")),
                broker_trade_id=payload.get("broker_trade_id"),
            )
            m = payload.get("metrics")
            if isinstance(m, dict):
                for k in self.metrics:
                    if k in m:
                        self.metrics[k] = int(m[k])
            return True
        except Exception as exc:
            self.logger.log("state_restore_failed", {"error": str(exc)})
            self.position = None
            return False

    @property
    def is_flat(self) -> bool:
        return self.position is None

    def on_candle_closed(
        self,
        ohlcv: pd.DataFrame,
        atr: pd.Series,
        bar_index: int,
        new_setups: list[TradeSetup],
    ) -> None:
        ts = ohlcv.index[bar_index]

        if self.position is not None:
            st, res = phase4_live_step(self.position.state, ohlcv, atr, bar_index)
            self.position.state = st
            if res is not None:
                tid = self.position.broker_trade_id
                self.logger.log(
                    "position_exit",
                    {
                        "exit_ts": str(res.exit_ts),
                        "outcome": res.outcome,
                        "realized_r": res.realized_r,
                        "entry_ts": str(self.position.setup.entry_ts),
                        "direction": self.position.setup.direction,
                        "broker_trade_id": tid,
                    },
                )
                self.broker.close_open_trade(tid)
                self.position = None
                self.metrics["close_events"] += 1

        if self.position is None and new_setups:
            self.metrics["detected_setups"] += len(new_setups)
            setup = new_setups[0]
            if len(new_setups) > 1:
                self.logger.log(
                    "signal_multi",
                    {"count": len(new_setups), "taking": "first", "ts": str(ts)},
                )
            plan = LiveExecutionPlan.from_setup(setup, ohlcv, atr, self.execution)
            state = phase4_live_init(ohlcv, setup, self.execution, atr)
            if state.closed or state.risk0 <= 1e-12:
                self.logger.log(
                    "setup_skip_invalid_risk",
                    {"entry_ts": str(setup.entry_ts), "direction": setup.direction},
                )
                return
            oid = f"{setup.session_date}_{setup.direction}_{pd.Timestamp(ts).value}"
            units = self.broker.suggest_units(plan)
            res_br = self.broker.place_market_entry_with_bracket(
                plan, units=units, client_order_id=oid
            )
            

        self.metrics["submitted_intents"] += 1
        self.logger.log(
            "order_intent",
            {
                "accepted": res_br.accepted,
                "client_order_id": res_br.client_order_id,
                "plan": plan,
                "broker_message": res_br.message,
                "broker_trade_id": res_br.broker_trade_id,
            },
        )
        if not res_br.accepted:
            self.logger.log(
                "order_rejected_skip",
                {"entry_ts": str(setup.entry_ts), "direction": setup.direction},
            )
            return
            st2, res_immediate = phase4_live_step(state, ohlcv, atr, bar_index)
            if res_immediate is not None:
                self.logger.log(
                    "position_exit_same_bar",
                    {
                        "exit_ts": str(res_immediate.exit_ts),
                        "outcome": res_immediate.outcome,
                        "realized_r": res_immediate.realized_r,
                        "broker_trade_id": res_br.broker_trade_id,
                    },
                )
                self.broker.close_open_trade(res_br.broker_trade_id)
                self.position = None
            else:
                self.position = ManagedPosition(
                    setup=setup,
                    plan=plan,
                    state=st2,
                    client_order_id=res_br.client_order_id,
                    broker_trade_id=res_br.broker_trade_id,
                )
                self.logger.log(
                    "position_open",
                    {
                        "entry_ts": str(setup.entry_ts),
                        "direction": setup.direction,
                        "sl": plan.stop_price,
                        "tp": plan.take_profit_price,
                        "risk_pips": plan.risk_pips,
                    },
                )
                self.metrics["open_events"] += 1


def _setup_to_dict(s: TradeSetup) -> dict:
    d = asdict(s)
    d["session_date"] = s.session_date.isoformat()
    d["sweep_ts"] = pd.Timestamp(s.sweep_ts).isoformat()
    d["confirm_ts"] = pd.Timestamp(s.confirm_ts).isoformat()
    d["entry_ts"] = pd.Timestamp(s.entry_ts).isoformat()
    return d


def _setup_from_dict(d: dict) -> TradeSetup:
    return TradeSetup(
        symbol=str(d["symbol"]),
        session_date=date.fromisoformat(str(d["session_date"])),
        direction=str(d["direction"]),
        sweep_ts=pd.Timestamp(d["sweep_ts"]),
        confirm_ts=pd.Timestamp(d["confirm_ts"]),
        entry_ts=pd.Timestamp(d["entry_ts"]),
        entry=float(d["entry"]),
        stop=float(d["stop"]),
        take_profit=float(d["take_profit"]),
        rr_target=float(d["rr_target"]),
        risk_points=float(d["risk_points"]),
        pip_or_point_size=float(d["pip_or_point_size"]),
        risk_in_pips=float(d["risk_in_pips"]),
        asian_high=float(d["asian_high"]),
        asian_low=float(d["asian_low"]),
    )


def _plan_to_dict(plan: LiveExecutionPlan) -> dict:
    d = asdict(plan)
    d["session_date"] = plan.session_date.isoformat()
    d["entry_ts"] = pd.Timestamp(plan.entry_ts).isoformat()
    d["sweep_ts"] = pd.Timestamp(plan.sweep_ts).isoformat()
    d["confirm_ts"] = pd.Timestamp(plan.confirm_ts).isoformat()
    return d


def _plan_from_dict(d: dict) -> LiveExecutionPlan:
    return LiveExecutionPlan(
        symbol=str(d["symbol"]),
        session_date=date.fromisoformat(str(d["session_date"])),
        direction=str(d["direction"]),
        entry_ts=pd.Timestamp(d["entry_ts"]),
        sweep_ts=pd.Timestamp(d["sweep_ts"]),
        confirm_ts=pd.Timestamp(d["confirm_ts"]),
        asian_high=float(d["asian_high"]),
        asian_low=float(d["asian_low"]),
        entry_price=float(d["entry_price"]),
        stop_price=float(d["stop_price"]),
        take_profit_price=float(d["take_profit_price"]),
        risk_points=float(d["risk_points"]),
        risk_pips=float(d["risk_pips"]),
        rr_target=float(d["rr_target"]),
        spread_pips=float(d["spread_pips"]),
        max_loss_cap_pips=float(d["max_loss_cap_pips"]),
    )


def _state_to_dict(s: Phase4LiveState) -> dict:
    d = asdict(s)
    d["setup"] = _setup_to_dict(s.setup)
    d["v"] = asdict(s.v)
    return d


def _state_from_dict(d: dict) -> Phase4LiveState:
    return Phase4LiveState(
        setup=_setup_from_dict(d["setup"]),
        v=ExecutionVariant(**d["v"]),
        start_i=int(d["start_i"]),
        entry=float(d["entry"]),
        risk0=float(d["risk0"]),
        sl_live=float(d["sl_live"]),
        tp_live=float(d["tp_live"]) if d.get("tp_live") is not None else None,
        pos=float(d["pos"]),
        total_r=float(d["total_r"]),
        be_armed=bool(d["be_armed"]),
        partial_done=bool(d["partial_done"]),
        one_r_px=float(d["one_r_px"]),
        mfe_r=float(d["mfe_r"]),
        closed=bool(d.get("closed", False)),
    )
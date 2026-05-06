"""
Live institutional EURUSD stack: ingestion, Asian/MSS signals, Phase 4 trade management.

Run CSV replay::

    python -m live.run_live_replay

Run OANDA-backed daemon (practice/live via ``OANDA_ENV``)::

    export OANDA_API_TOKEN=... OANDA_ACCOUNT_ID=...
    python -m live.run_daemon --paper   # live feed, paper orders
    python -m live.run_daemon           # live feed + OANDA execution

Wire other venues by implementing ``live.broker.base.BrokerClient``.
"""

from live.broker import BrokerClient, BrokerOrderResult, PaperBrokerClient
from live.candle_buffer import CandleBuffer
from live.execution_plan import LiveExecutionPlan
from live.ingestion import BarSource, CsvReplaySource, WebSocketBarSourceStub
from live.runner import LiveEURUSDBot, default_replay_bot
from live.signal_engine import AsianSessionSnapshot, InstitutionalEURUSDSignalEngine
from live.trade_logger import TradeLogger
from live.trade_manager import ManagedPosition, QuantTradeManager
from live.types import Candle

__all__ = [
    "AsianSessionSnapshot",
    "BarSource",
    "BrokerClient",
    "BrokerOrderResult",
    "Candle",
    "CandleBuffer",
    "CsvReplaySource",
    "InstitutionalEURUSDSignalEngine",
    "LiveEURUSDBot",
    "LiveExecutionPlan",
    "ManagedPosition",
    "PaperBrokerClient",
    "QuantTradeManager",
    "TradeLogger",
    "WebSocketBarSourceStub",
    "default_replay_bot",
]

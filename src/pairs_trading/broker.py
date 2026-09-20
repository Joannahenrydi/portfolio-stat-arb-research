from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TargetOrder:
    symbol: str
    qty: int


class AlpacaPaperBroker:
    """Deliberately paper-only adapter; no live mode exists in this project."""

    def __init__(self) -> None:
        try:
            from alpaca.data.historical import StockHistoricalDataClient
            from alpaca.trading.client import TradingClient
        except ImportError as exc:
            raise RuntimeError("install with: pip install -e '.[alpaca]'") from exc
        key = os.environ.get("APCA_API_KEY_ID")
        secret = os.environ.get("APCA_API_SECRET_KEY")
        if not key or not secret:
            raise RuntimeError("APCA_API_KEY_ID and APCA_API_SECRET_KEY are required")
        self.trading = TradingClient(key, secret, paper=True)
        self.data = StockHistoricalDataClient(key, secret)

    def daily_prices(self, symbols: list[str], lookback_days: int) -> pd.DataFrame:
        from alpaca.data.enums import Adjustment, DataFeed
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame

        end = pd.Timestamp.now(tz="America/New_York").normalize().to_pydatetime()
        req = StockBarsRequest(
            symbol_or_symbols=symbols,
            timeframe=TimeFrame.Day,
            feed=DataFeed.IEX,
            adjustment=Adjustment.ALL,
            end=end - timedelta(microseconds=1),
            start=datetime.now(timezone.utc) - timedelta(days=lookback_days),
        )
        bars = self.data.get_stock_bars(req).df.reset_index()
        return bars.pivot(index="timestamp", columns="symbol", values="close").dropna()

    def submit_market_orders(self, orders: list[TargetOrder]) -> None:
        if os.environ.get("ALLOW_PAPER_ORDERS", "false").lower() != "true":
            print("DRY RUN:", orders)
            return
        from alpaca.trading.enums import OrderSide, TimeInForce
        from alpaca.trading.requests import MarketOrderRequest

        submitted = []
        try:
            for order in orders:
                request = MarketOrderRequest(
                    symbol=order.symbol,
                    qty=abs(order.qty),
                    side=OrderSide.BUY if order.qty > 0 else OrderSide.SELL,
                    time_in_force=TimeInForce.DAY,
                )
                submitted.append(self.trading.submit_order(order_data=request))
        except Exception:
            for item in submitted:
                try:
                    self.trading.cancel_order_by_id(item.id)
                except Exception as cancel_error:  # noqa: BLE001
                    logger.warning("failed to cancel order %s: %s", item.id, cancel_error)
            raise

    def pair_position(self, x: str, y: str) -> int:
        """Return +1 long spread, -1 short spread, 0 flat; reject orphan legs."""
        quantities = {p.symbol: float(p.qty) for p in self.trading.get_all_positions()}
        qx, qy = quantities.get(x, 0.0), quantities.get(y, 0.0)
        if qx == 0 and qy == 0:
            return 0
        if qx == 0 or qy == 0 or qx * qy >= 0:
            raise RuntimeError(f"unsafe/orphan pair position: {x}={qx}, {y}={qy}; reconcile manually")
        return 1 if qy > 0 else -1

    def close_pair(self, x: str, y: str) -> None:
        if os.environ.get("ALLOW_PAPER_ORDERS", "false").lower() != "true":
            print("DRY RUN: close", x, y)
            return
        # Alpaca closes each complete symbol position; this stays paper-only.
        self.trading.close_position(x)
        self.trading.close_position(y)

"""Portfolio-level Alpaca paper planning and fail-closed submission."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PaperOrder:
    symbol: str
    quantity: int
    reference_price: float
    target_weight: float


PAPER_BASE_URL = "https://paper-api.alpaca.markets"


def build_order_plan(
    target_weights: pd.Series,
    current_shares: pd.Series,
    prices: pd.Series,
    equity: float,
    *,
    orders_allowed: bool,
) -> list[PaperOrder]:
    """Convert portfolio weights to whole-share deltas after a promotion gate."""
    if not orders_allowed:
        return []
    index = target_weights.index
    if not index.is_unique or set(current_shares.index) != set(index) or set(prices.index) != set(index):
        raise ValueError("targets, holdings, and prices must share unique symbols")
    current = current_shares.reindex(index).astype(float)
    marks = prices.reindex(index).astype(float)
    targets = target_weights.astype(float)
    if not np.isfinite(equity) or equity <= 0:
        raise ValueError("equity must be finite and positive")
    if not np.isfinite(marks).all() or (marks <= 0).any() or not np.isfinite(targets).all():
        raise ValueError("prices and target weights must be finite; prices must be positive")
    desired = (targets * equity / marks).round().astype(int)
    delta = desired - current.round().astype(int)
    return [
        PaperOrder(symbol, int(delta[symbol]), float(marks[symbol]), float(targets[symbol]))
        for symbol in index
        if delta[symbol] != 0
    ]


def submit_paper_orders(
    orders: list[PaperOrder],
    *,
    orders_allowed: bool,
    submission_enabled: bool | None = None,
) -> list[dict]:
    """Submit market-day orders exclusively to Alpaca's paper endpoint.

    Promotion must allow orders and ``ALPACA_PAPER_SUBMIT=YES`` must be set (or
    ``submission_enabled=True`` supplied by an explicit caller). Credentials are
    read only from the process environment and are never written to an artifact.
    """
    enabled = (
        os.environ.get("ALPACA_PAPER_SUBMIT") == "YES"
        if submission_enabled is None
        else submission_enabled
    )
    if not orders_allowed or not enabled:
        return []
    key = os.environ.get("APCA_API_KEY_ID")
    secret = os.environ.get("APCA_API_SECRET_KEY")
    if not key or not secret:
        raise RuntimeError("Alpaca credentials are required in the process environment")
    responses = []
    session_tag = datetime.now(timezone.utc).strftime("%Y%m%d")
    for position, order in enumerate(orders):
        if order.quantity == 0:
            continue
        payload = {
            "symbol": order.symbol,
            "qty": str(abs(order.quantity)),
            "side": "buy" if order.quantity > 0 else "sell",
            "type": "market",
            "time_in_force": "day",
            "client_order_id": f"mn-{session_tag}-{position}-{order.symbol}"[:48],
        }
        request = Request(
            f"{PAPER_BASE_URL}/v2/orders",
            data=json.dumps(payload).encode(),
            method="POST",
            headers={
                "Content-Type": "application/json",
                "APCA-API-KEY-ID": key,
                "APCA-API-SECRET-KEY": secret,
            },
        )
        try:
            with urlopen(request, timeout=30) as response:
                result = json.loads(response.read())
        except HTTPError as exc:
            detail = exc.read().decode(errors="replace")
            raise RuntimeError(
                f"Alpaca paper order failed: HTTP {exc.code}; request_id="
                f"{exc.headers.get('X-Request-ID')}; {detail}"
            ) from None
        responses.append(
            {
                "symbol": order.symbol,
                "requested_quantity": order.quantity,
                "target_weight": order.target_weight,
                "order_id": result.get("id"),
                "client_order_id": result.get("client_order_id"),
                "status": result.get("status"),
                "submitted_at": result.get("submitted_at"),
                "filled_quantity": result.get("filled_qty"),
                "filled_average_price": result.get("filled_avg_price"),
            }
        )
    return responses

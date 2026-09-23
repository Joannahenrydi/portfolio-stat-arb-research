"""Portfolio PnL attribution with explicit modeled costs."""

from __future__ import annotations

import numpy as np
import pandas as pd


def attribute_pnl(
    prior_weights: pd.Series,
    realized_returns: pd.Series,
    *,
    transaction_cost: float = 0.0,
    borrow_cost: float = 0.0,
) -> tuple[pd.Series, dict[str, float]]:
    """Return security contributions and a reconciled portfolio return ledger."""
    if not prior_weights.index.is_unique or set(prior_weights.index) != set(realized_returns.index):
        raise ValueError("weights and returns must share unique security IDs")
    weights = prior_weights.astype(float)
    returns = realized_returns.reindex(weights.index).astype(float)
    if not np.isfinite(weights).all() or not np.isfinite(returns).all():
        raise ValueError("held weights and realized returns must be finite")
    if min(transaction_cost, borrow_cost) < 0 or not np.isfinite([transaction_cost, borrow_cost]).all():
        raise ValueError("costs must be finite and nonnegative")
    contributions = (weights * returns).rename("gross_contribution")
    gross = float(contributions.sum())
    ledger = {
        "gross_return": gross,
        "transaction_cost": float(transaction_cost),
        "borrow_cost": float(borrow_cost),
        "net_return": gross - transaction_cost - borrow_cost,
    }
    return contributions, ledger

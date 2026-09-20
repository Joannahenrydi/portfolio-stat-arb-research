from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .model import StrategyConfig, build_signals


@dataclass
class BacktestResult:
    equity: pd.DataFrame
    trades: pd.DataFrame
    signals: pd.DataFrame
    summary: dict[str, float | int]


def run_backtest(
    prices: pd.DataFrame, x: str, y: str, cfg: StrategyConfig,
    trade_start: str | None = None,
) -> BacktestResult:
    sig = build_signals(prices, x, y, cfg)
    start = pd.Timestamp(trade_start) if trade_start else sig.index[cfg.formation_window]
    if not (sig.index >= start).any():
        raise ValueError("trade_start is after the available data")
    cash = cfg.initial_capital
    qx = qy = 0
    pending = 0
    position = 0
    held = 0
    entry_equity = cash
    records, trades = [], []
    prev_px = prev_py = None
    cost_rate = (cfg.commission_bps + cfg.slippage_bps) / 10_000

    for date, row in sig.iterrows():
        px, py, z = float(row[x]), float(row[y]), float(row.zscore)
        if prev_px is not None:
            cash += qx * (px - prev_px) + qy * (py - prev_py)

        # Execute yesterday's decision at today's bar, avoiding same-close lookahead.
        if pending and position == 0:
            gross = max(cash, 0) * cfg.gross_pair_fraction
            leg = gross / 2
            qy = int(leg / py) * pending
            qx = -int(leg / px) * pending
            if qx and qy:
                fee = (abs(qx) * px + abs(qy) * py) * cost_rate
                cash -= fee
                position, held, entry_equity = pending, 0, cash
                trades.append({"date": date, "event": "ENTRY", "side": position, "z": z, "fee": fee})
            else:
                qx = qy = 0
            pending = 0
        elif pending and position != 0:
            fee = (abs(qx) * px + abs(qy) * py) * cost_rate
            cash -= fee
            trades.append({"date": date, "event": "EXIT", "side": position, "z": z, "fee": fee})
            qx = qy = position = held = pending = 0

        equity = cash
        if position:
            held += 1
            should_exit = (
                (position == 1 and z >= -cfg.exit_z)
                or (position == -1 and z <= cfg.exit_z)
                or abs(z) >= cfg.stop_z
                or held >= cfg.max_holding_bars
                or equity <= entry_equity * 0.95
            )
            if should_exit:
                pending = 2  # any nonzero value triggers a close next bar
        elif date >= start and np.isfinite(z):
            if z <= -cfg.entry_z:
                pending = 1
            elif z >= cfg.entry_z:
                pending = -1

        records.append({"date": date, "equity": equity, "position": position, "qx": qx, "qy": qy})
        prev_px, prev_py = px, py

    equity_df = pd.DataFrame(records).set_index("date")
    equity_df = equity_df.loc[equity_df.index >= start]
    returns = equity_df.equity.pct_change().fillna(0)
    drawdown = equity_df.equity / equity_df.equity.cummax() - 1
    ann_vol = float(returns.std(ddof=1) * np.sqrt(252))
    summary = {
        "initial_capital": cfg.initial_capital,
        "final_equity": float(equity_df.equity.iloc[-1]),
        "total_return": float(equity_df.equity.iloc[-1] / cfg.initial_capital - 1),
        "annualized_volatility": ann_vol,
        "sharpe_zero_rf": float(returns.mean() * 252 / ann_vol) if ann_vol else 0.0,
        "max_drawdown": float(drawdown.min()),
        "entries": sum(t["event"] == "ENTRY" for t in trades),
        "exits": sum(t["event"] == "EXIT" for t in trades),
        "fees_and_slippage": float(sum(t["fee"] for t in trades)),
        "open_position_at_end": int(position),
        "evaluation_bars": len(equity_df),
    }
    return BacktestResult(equity_df, pd.DataFrame(trades), sig, summary)

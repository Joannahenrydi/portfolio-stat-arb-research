"""Cost-aware research engine, separate from the paper broker.

Uses actual close prices and dividend cash flows. Datasets with stock splits
are rejected: a split-aware execution ledger is not implemented here.
"""
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .model import StrategyConfig, build_signals


@dataclass(frozen=True)
class Costs:
    commission_per_share: float = 0.005
    minimum_per_order: float = 1.0
    slippage_bps: float = 2.0
    sell_fee_bps: float = 0.3
    borrow_annual: float = 0.03

    def execution(self, quantities, prices):
        commission = slippage = 0.0
        for qty, price in zip(quantities, prices):
            if qty:
                commission += max(self.minimum_per_order, abs(qty) * self.commission_per_share)
                commission += max(-qty, 0) * price * self.sell_fee_bps / 10000
                slippage += abs(qty) * price * self.slippage_bps / 10000
        return commission, slippage


@dataclass(frozen=True)
class Parameters:
    model: str = "kalman"
    window: int = 60
    process_variance: float = 1e-5
    entry: float = 1.5
    exit: float = 0.25
    stop: float = 4.0
    holding: int = 20
    reverting_only: bool = False
    cost_buffer: float = 1.5


def load_market(directory, symbols=None):
    numbers = [0, 1]
    if symbols is not None:
        if len(symbols) != 2 or symbols[0] == symbols[1]:
            raise ValueError("exactly two distinct symbols required")
        metadata = json.loads((Path(directory) / "metadata.json").read_text())
        numbers = [metadata["symbols"].index(symbol) for symbol in symbols]
    legs = []
    for number, symbol in zip(numbers, ("X", "Y")):
        frame = pd.read_csv(Path(directory) / f"raw_{number}.csv")
        if (frame["Stock Splits"].fillna(0) != 0).any():
            raise ValueError("split events require a split-aware ledger; dataset rejected")
        frame.index = pd.to_datetime(frame.iloc[:, 0].str.slice(0, 10))
        dividends = frame["Dividends"].fillna(0) + frame.get("Capital Gains", 0)
        leg = pd.DataFrame({symbol: frame.Close, "div_" + symbol: dividends})
        if leg.index.has_duplicates:
            raise ValueError("duplicate market dates")
        legs.append(leg)
    market = pd.concat(legs, axis=1).sort_index().dropna()
    if not np.isfinite(market.to_numpy()).all() or (market[["X", "Y"]] <= 0).any().any():
        raise ValueError("invalid market prices")
    return market


def total_return_prices(market):
    result = pd.DataFrame(index=market.index)
    for symbol in ("X", "Y"):
        growth = (market[symbol] + market["div_" + symbol]) / market[symbol].shift(1)
        result[symbol] = growth.fillna(1).cumprod() * market[symbol].iloc[0]
    return result


def research_signals(market, params):
    prices = total_return_prices(market)
    if params.model == "kalman":
        cfg = StrategyConfig(z_window=params.window, process_variance=params.process_variance)
        result = build_signals(prices, "X", "Y", cfg)
        spread = result.spread
        mean = spread.shift(1).rolling(params.window).mean()
        deviation = spread - mean
        beta = result.beta
        z = result.zscore
    elif params.model == "rolling_ols":
        # All fit statistics exclude the current observation.
        x, y = prices.X.shift(1), prices.Y.shift(1)
        beta = x.rolling(params.window).cov(y) / x.rolling(params.window).var()
        alpha = y.rolling(params.window).mean() - beta * x.rolling(params.window).mean()
        spread = prices.Y - alpha - beta * prices.X
        deviation = spread - spread.shift(1).rolling(params.window).mean()
        z = deviation / spread.shift(1).rolling(params.window).std()
    else:
        raise ValueError("unknown model")
    # Convert total-return-index hedge units to actual share quantities.
    hedge = beta * (prices.X / market.X) / (prices.Y / market.Y)
    edge_per_y_share = deviation.abs() / (prices.Y / market.Y)
    return pd.DataFrame({"z": z, "hedge": hedge, "edge": edge_per_y_share}, index=market.index)


def simulate(market, signals, params, start, end, fraction=0.2, capital=100000, costs=None):
    costs = costs if costs is not None else Costs()
    if not 0 <= fraction <= 1 or capital <= 0:
        raise ValueError("fraction must be in [0, 1] and capital positive")
    if any(value < 0 for value in vars(costs).values()):
        raise ValueError("costs cannot be negative")
    data = market.join(signals).loc[start:end]
    if len(data) < 2:
        raise ValueError("evaluation requires at least two observations")
    values = data[["X", "Y", "div_X", "div_Y", "z", "hedge", "edge"]].to_numpy()
    account = capital
    qx = qy = direction = holding = 0
    pending = None
    logs, records = [], []
    commission_total = slippage_total = borrow_total = dividends_total = 0.0
    for i, (date, row) in enumerate(zip(data.index, values)):
        px, py, dx, dy, z, hedge, edge = row
        if i:
            prev = values[i - 1]
            days = (date - data.index[i - 1]).days
            dividend = qx * dx + qy * dy
            borrow = (max(-qx, 0) * prev[0] + max(-qy, 0) * prev[1]) * costs.borrow_annual * days / 365
            account += qx * (px - prev[0]) + qy * (py - prev[1]) + dividend - borrow
            dividends_total += dividend
            borrow_total += borrow
        if pending is not None:
            action, requested_direction, requested_hedge = pending
            if action == "exit":
                commission, slip = costs.execution((-qx, -qy), (px, py))
                account -= commission + slip
                logs.append({"date": date, "event": "EXIT", "qx": -qx, "qy": -qy,
                             "px": px, "py": py, "commission": commission, "slippage": slip})
                qx = qy = direction = holding = 0
            else:
                gross = max(account, 0) * fraction
                units = int(gross / (py + requested_hedge * px))
                new_y = units * requested_direction
                new_x = -int(units * requested_hedge) * requested_direction
                commission = slip = 0.0
                if new_x and new_y:
                    qx, qy, direction = new_x, new_y, requested_direction
                    commission, slip = costs.execution((qx, qy), (px, py))
                    account -= commission + slip
                    logs.append({"date": date, "event": "ENTRY", "qx": qx, "qy": qy,
                                 "px": px, "py": py, "hedge": requested_hedge,
                                 "commission": commission, "slippage": slip})
            commission_total += commission
            slippage_total += slip
            pending = None
        # Scheduled end-of-experiment liquidation, including both leg costs.
        if i == len(data) - 1 and direction:
            commission, slip = costs.execution((-qx, -qy), (px, py))
            account -= commission + slip
            commission_total += commission
            slippage_total += slip
            logs.append({"date": date, "event": "FINAL_EXIT", "qx": -qx, "qy": -qy,
                         "px": px, "py": py, "commission": commission, "slippage": slip})
            qx = qy = direction = 0
        if direction:
            holding += 1
            if (not np.isfinite(z) or direction * z >= -params.exit
                    or abs(z) >= params.stop or holding >= params.holding):
                pending = ("exit", 0, 0)
        elif i < len(data) - 2 and fraction and np.isfinite([z, hedge, edge]).all():
            if hedge > 0 and params.entry <= abs(z) < params.stop:
                reverting = i > 0 and np.isfinite(values[i-1, 4]) and abs(z) < abs(values[i-1, 4])
                if not params.reverting_only or reverting:
                    units = int(max(account, 0) * fraction / (py + hedge * px))
                    side = 1 if z < 0 else -1
                    tx, ty = -int(units * hedge) * side, units * side
                    entry_cost = sum(costs.execution((tx, ty), (px, py)))
                    exit_cost = sum(costs.execution((-tx, -ty), (px, py)))
                    # Expected holding duration is a conservative fixed screening assumption.
                    borrow_estimate = (max(-tx, 0) * px + max(-ty, 0) * py) * costs.borrow_annual * params.holding * 1.5 / 365
                    if tx and ty and units * edge > params.cost_buffer * (entry_cost + exit_cost + borrow_estimate):
                        pending = ("entry", side, hedge)
        records.append({"date": date, "equity": account, "qx": qx, "qy": qy,
                        "gross": abs(qx) * px + abs(qy) * py, "net": qx * px + qy * py})
    curve = pd.DataFrame(records).set_index("date")
    returns = curve.equity.pct_change().fillna(0)
    volatility = float(returns.std() * np.sqrt(252))
    summary = {
        "return": float(account / capital - 1), "final_equity": float(account),
        "drawdown": float((curve.equity / curve.equity.cummax() - 1).min()),
        "sharpe": float(returns.mean() * 252 / volatility) if volatility else 0.0,
        "entries": sum(t["event"] == "ENTRY" for t in logs),
        "commission": commission_total, "slippage": slippage_total,
        "borrow": borrow_total, "dividends_net": dividends_total,
        "max_gross": float(curve.gross.max()), "max_abs_net": float(curve.net.abs().max()),
    }
    return summary, curve, pd.DataFrame(logs)

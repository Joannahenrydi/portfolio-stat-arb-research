from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .backtest import run_backtest
from .broker import AlpacaPaperBroker, TargetOrder
from .model import StrategyConfig, build_signals
from .selection import select_pairs


def load_config(path: str | None) -> StrategyConfig:
    if not path:
        return StrategyConfig()
    return StrategyConfig(**json.loads(Path(path).read_text()))


def synthetic_prices(n: int = 900, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2021-01-01", periods=n)
    x = 80 + np.cumsum(rng.normal(0.04, 0.7, n))
    spread = np.zeros(n)
    for i in range(1, n):
        spread[i] = 0.90 * spread[i - 1] + rng.normal(0, 0.8)
    beta = 1.15 + np.cumsum(rng.normal(0, 0.00015, n))
    y = beta * x + spread
    return pd.DataFrame({"PAIR_X": x, "PAIR_Y": y}, index=dates)


def save_result(result, output: str) -> None:
    path = Path(output)
    path.mkdir(parents=True, exist_ok=True)
    result.equity.to_csv(path / "equity.csv")
    result.trades.to_csv(path / "trades.csv", index=False)
    result.signals.to_csv(path / "signals.csv")
    (path / "summary.json").write_text(json.dumps(result.summary, indent=2) + "\n")
    print(json.dumps(result.summary, indent=2))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Paper-first Kalman pairs trading")
    sub = parser.add_subparsers(dest="command", required=True)
    demo = sub.add_parser("demo")
    demo.add_argument("--output", default="output/demo")
    demo.add_argument("--config")
    backtest = sub.add_parser("backtest")
    backtest.add_argument("--csv", required=True)
    backtest.add_argument("--x", required=True)
    backtest.add_argument("--y", required=True)
    backtest.add_argument("--output", default="output/backtest")
    backtest.add_argument("--config")
    backtest.add_argument("--trade-start", help="first eligible signal date; earlier rows warm up only")
    download = sub.add_parser("download", help="fetch real adjusted daily prices from Yahoo Finance")
    download.add_argument("--symbols", nargs="+", required=True)
    download.add_argument("--start", required=True)
    download.add_argument("--end", required=True, help="exclusive end date")
    download.add_argument("--output", default="output/market_data")
    alpaca = sub.add_parser("download-alpaca", help="fetch explicit-feed Alpaca daily research bars")
    alpaca.add_argument("--symbols", nargs="+", required=True)
    alpaca.add_argument("--start", required=True)
    alpaca.add_argument("--end", required=True, help="exclusive end, New York time")
    alpaca.add_argument("--feed", choices=["iex", "sip"], default="iex")
    alpaca.add_argument("--output", default="output/alpaca/market_data")
    select = sub.add_parser("select")
    select.add_argument("--csv", required=True)
    select.add_argument("--skip-i1-check", action="store_true")
    paper = sub.add_parser("paper")
    paper.add_argument("--x", required=True)
    paper.add_argument("--y", required=True)
    paper.add_argument("--lookback-days", type=int, default=400)
    paper.add_argument("--config")
    args = parser.parse_args(argv)
    cfg = load_config(getattr(args, "config", None))

    if args.command == "demo":
        save_result(run_backtest(synthetic_prices(), "PAIR_X", "PAIR_Y", cfg), args.output)
    elif args.command == "backtest":
        prices = pd.read_csv(args.csv, index_col=0, parse_dates=True)
        save_result(run_backtest(prices, args.x, args.y, cfg, trade_start=args.trade_start), args.output)
    elif args.command == "download":
        from .data import download_daily

        prices = download_daily(args.symbols, args.start, args.end, args.output)
        print(json.dumps({"rows": len(prices), "output": args.output}))
    elif args.command == "download-alpaca":
        from .alpaca_data import download_alpaca

        print(json.dumps(download_alpaca(args.symbols, args.start, args.end,
                                        args.output, args.feed), indent=2))
    elif args.command == "select":
        prices = pd.read_csv(args.csv, index_col=0, parse_dates=True)
        print(json.dumps(select_pairs(prices, require_i1=not args.skip_i1_check)))
    else:
        broker = AlpacaPaperBroker()
        prices = broker.daily_prices([args.x, args.y], args.lookback_days)
        latest_time = pd.Timestamp(prices.index[-1])
        if latest_time.tzinfo is None:
            latest_time = latest_time.tz_localize("UTC")
        age_hours = (datetime.now(timezone.utc) - latest_time.to_pydatetime()).total_seconds() / 3600
        if age_hours > cfg.max_data_age_hours:
            raise RuntimeError(f"latest complete bar is stale ({age_hours:.1f} hours old)")
        signals = build_signals(prices, args.x, args.y, cfg)
        latest = signals.iloc[-1]
        z = float(latest.zscore)
        current = broker.pair_position(args.x, args.y)
        exit_now = current and (
            (current == 1 and z >= -cfg.exit_z)
            or (current == -1 and z <= cfg.exit_z)
            or abs(z) >= cfg.stop_z
        )
        if exit_now:
            broker.close_pair(args.x, args.y)
            print(json.dumps({"action": "EXIT", "position": current, "zscore": z}))
            return
        if current or not np.isfinite(z) or abs(z) < cfg.entry_z:
            print(json.dumps({"action": "HOLD", "position": current, "zscore": z, "config": asdict(cfg)}, default=str))
            return
        account = broker.trading.get_account()
        gross = float(account.equity) * cfg.gross_pair_fraction
        direction = 1 if z <= -cfg.entry_z else -1
        orders = [
            TargetOrder(args.y, direction * int((gross / 2) / float(latest[args.y]))),
            TargetOrder(args.x, -direction * int((gross / 2) / float(latest[args.x]))),
        ]
        if all(order.qty != 0 for order in orders):
            broker.submit_market_orders(orders)


if __name__ == "__main__":
    main()

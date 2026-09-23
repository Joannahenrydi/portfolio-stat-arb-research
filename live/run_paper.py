"""Run a portfolio order file through the fail-closed Alpaca paper adapter."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from live.paper import build_order_plan, submit_paper_orders


def main(
    decision_path: Path,
    targets_path: Path,
    positions_path: Path,
    prices_path: Path,
    equity: float,
    output_path: Path,
) -> None:
    decision = json.loads(decision_path.read_text())
    allowed = decision.get("status") == "ACCEPTED" and decision.get("orders_allowed") is True
    targets = pd.read_csv(targets_path).set_index("symbol")["target_weight"]
    positions = pd.read_csv(positions_path).set_index("symbol")["shares"].reindex(targets.index, fill_value=0)
    prices = pd.read_csv(prices_path).set_index("symbol")["price"].reindex(targets.index)
    plan = build_order_plan(targets, positions, prices, equity, orders_allowed=allowed)
    responses = submit_paper_orders(plan, orders_allowed=allowed)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            {
                "promotion_status": decision.get("status"),
                "orders_allowed": allowed,
                "planned_orders": [order.__dict__ for order in plan],
                "broker_responses": responses,
            },
            indent=2,
        )
        + "\n"
    )


def cli() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--decision", type=Path, required=True)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--positions", type=Path, required=True)
    parser.add_argument("--prices", type=Path, required=True)
    parser.add_argument("--equity", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    main(args.decision, args.targets, args.positions, args.prices, args.equity, args.output)


if __name__ == "__main__":
    cli()

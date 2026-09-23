"""Generate a local shadow target snapshot; this script never submits an order."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def main(decision_path: Path, target_path: Path, data_audit_path: Path, output: Path) -> None:
    decision = json.loads(decision_path.read_text())
    data_audit = json.loads(data_audit_path.read_text())
    targets = pd.read_csv(target_path)
    last_bar = pd.Timestamp(data_audit["last_bar_session"]).date()
    age_days = (datetime.now(timezone.utc).date() - last_bar).days
    promoted = decision.get("status") == "ACCEPTED" and decision.get("orders_allowed") is True
    fresh = age_days <= 4
    status = "READY" if promoted and fresh else "BLOCKED"
    reasons = []
    if not promoted:
        reasons.append("PROMOTION_DECISION_REJECTED")
    if not fresh:
        reasons.append("MARKET_DATA_STALE")
    snapshot = targets[["symbol", "security_id", "sector", "target_weight"]].copy()
    snapshot = snapshot.rename(columns={"target_weight": "theoretical_weight"})
    snapshot["approved_weight"] = snapshot.theoretical_weight if status == "READY" else 0.0
    snapshot["order_authorized"] = False
    output.mkdir(parents=True, exist_ok=True)
    snapshot.to_csv(output / "shadow_targets.csv", index=False)
    event = {
        "status": status,
        "reasons": reasons,
        "last_bar_session": str(last_bar),
        "data_age_calendar_days": age_days,
        "theoretical_gross": float(snapshot.theoretical_weight.abs().sum()),
        "approved_gross": float(snapshot.approved_weight.abs().sum()),
        "submitted_orders": 0,
        "fills": 0,
        "failed_orders": 0,
    }
    (output / "shadow_event.json").write_text(json.dumps(event, indent=2) + "\n")
    print(json.dumps(event, indent=2))


def cli() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--decision", type=Path, default=Path("reports/equity_final/promotion_decision.json")
    )
    parser.add_argument(
        "--targets", type=Path, default=Path("reports/equity_v2/backtest/paper_targets_preview.csv")
    )
    parser.add_argument(
        "--data-audit", type=Path,
        default=Path("reports/equity_v2/data_quality/data_audit.json"),
    )
    parser.add_argument("--output", type=Path, default=Path("reports/equity_final/shadow_paper"))
    args = parser.parse_args()
    main(args.decision, args.targets, args.data_audit, args.output)


if __name__ == "__main__":
    cli()

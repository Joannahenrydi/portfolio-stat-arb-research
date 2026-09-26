"""Evaluate a time-series cross-asset trend sleeve under frozen v13 B03 budgets."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd

from features.alphas import forward_total_return_labels
from portfolio.optimizer import PortfolioCosts
from scripts.evaluate_cross_asset_v12 import TEST, TRAIN, VALIDATION, build_family_scores, calibrate, load_data
from scripts.evaluate_cross_asset_v13 import (
    asset_sleeves,
    normalized_factor_loadings,
    passes,
    run_budgeted,
    segment_metrics,
)
from scripts.evaluate_equity_v7 import sha256

PROTOCOL = Path("docs/MULTI_ASSET_PROTOCOL_V15.md")


def build_time_series_trend(data: dict) -> pd.DataFrame:
    log_price = np.log(data["close"])
    volatility = data["returns"].rolling(60, min_periods=60).std().shift(1)
    parts = []
    for horizon in (63, 126, 252):
        parts.append(
            log_price.shift(21).sub(log_price.shift(horizon)).div(
                volatility * np.sqrt(horizon - 21)
            )
        )
    return (sum(parts) / len(parts)).clip(-3, 3).where(data["eligibility"])


def calibrate_absolute(score: pd.DataFrame, returns: pd.DataFrame) -> dict:
    labels, label_end = forward_total_return_labels(returns, 5)
    completed = label_end.le(TRAIN[1])
    x = score.loc[TRAIN[0] : TRAIN[1]].where(completed.loc[TRAIN[0] : TRAIN[1]]).to_numpy().ravel()
    y = labels.loc[TRAIN[0] : TRAIN[1]].where(completed.loc[TRAIN[0] : TRAIN[1]]).to_numpy().ravel()
    usable = np.isfinite(x) & np.isfinite(y)
    if usable.sum() < 1000:
        return {"status": "BLOCKED", "reason": "INSUFFICIENT_CALIBRATION", "slope": np.nan}
    slope = float(x[usable] @ y[usable] / (x[usable] @ x[usable] + 1e-6))
    correlation = float(np.corrcoef(x[usable], y[usable])[0, 1])
    return {
        "status": "ADMITTED" if slope > 0 else "BLOCKED",
        "reason": "OK" if slope > 0 else "NONPOSITIVE_TRAIN_SLOPE",
        "slope": slope, "observations": int(usable.sum()), "pooled_correlation": correlation,
        "directional_accuracy": float((np.sign(x[usable]) == np.sign(y[usable])).mean()),
        "digest": hashlib.sha256(np.c_[x[usable], y[usable]].tobytes()).hexdigest(),
    }


def main(source: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    data = load_data(source, VALIDATION[1])
    cross_sectional = build_family_scores(data)["trend"]
    time_series = build_time_series_trend(data)
    artifacts = {
        "cross_sectional_trend": calibrate(cross_sectional, data["returns"]),
        "time_series_trend": calibrate_absolute(time_series, data["returns"]),
    }
    pd.DataFrame([{"family": name, **value} for name, value in artifacts.items()]).to_csv(
        output / "train_family_calibration.csv", index=False
    )
    if artifacts["time_series_trend"]["status"] != "ADMITTED":
        summary = {"status": "REJECTED", "reason": artifacts["time_series_trend"]["reason"],
                   "orders_allowed": False, "locked_test_evaluated": False}
        (output / "SUMMARY.json").write_text(json.dumps(summary, indent=2) + "\n")
        return
    scores = {
        "T00": cross_sectional * artifacts["cross_sectional_trend"]["slope"],
        "T01": time_series * artifacts["time_series_trend"]["slope"],
    }
    loadings = normalized_factor_loadings()
    sleeves = asset_sleeves(loadings.index)
    costs = PortfolioCosts(
        commission_bps=0, half_spread_bps=1, slippage_bps=1,
        impact_coefficient=.10, borrow_annual=.01, calendar_days=7,
    )
    freeze = {
        "protocol_sha256": sha256(PROTOCOL), "source_sha256": sha256(source),
        "artifacts": artifacts, "portfolio": "frozen_v13_B03", "costs": asdict(costs),
        "development_reused": True, "locked_test_loaded_for_selection": False,
    }
    (output / "frozen_run_inputs.json").write_text(json.dumps(freeze, indent=2) + "\n")

    rows = []
    for candidate in ("T00", "T01"):
        result = run_budgeted(scores[candidate], data, loadings, sleeves, "B03", costs)
        result.daily.to_csv(output / f"{candidate}_daily.csv")
        result.rebalances.to_csv(output / f"{candidate}_rebalances.csv")
        train, development = segment_metrics(result, *TRAIN), segment_metrics(result, *VALIDATION)
        selectable = candidate == "T01"
        eligible = selectable and passes(train, development, budgeted=True)
        row = {"experiment": candidate, "selectable": selectable, "eligible": eligible,
               "status": result.status, "reason": result.reason}
        row.update({f"train_{key}": value for key, value in train.items()})
        row.update({f"development_{key}": value for key, value in development.items()})
        rows.append(row)
    selection = pd.DataFrame(rows).sort_values(["eligible", "experiment"], ascending=[False, True])
    selection.to_csv(output / "candidate_selection.csv", index=False)
    selected = "T01" if bool(selection.set_index("experiment").loc["T01", "eligible"]) else None
    selection_lock = {
        "status": "SELECTED_PENDING_KILL_TESTS" if selected else "REJECTED",
        "reason": None if selected else "TIME_SERIES_TREND_FAILED_FROZEN_GATES",
        "candidate": selected, "locked_test_evaluated": False,
        "selection_sha256": sha256(output / "candidate_selection.csv"),
    }
    (output / "selection_lock.json").write_text(json.dumps(selection_lock, indent=2) + "\n")

    kill_rows = []
    if selected:
        tests = [
            ("double_cost", scores[selected], 1.0, 5, replace(costs, multiplier=2.0)),
            ("signal_delay_1", scores[selected].shift(1), 1.0, 5, costs),
            ("caps_tighten_20pct", scores[selected], .8, 5, costs),
            ("caps_loosen_20pct", scores[selected], 1.2, 5, costs),
            ("rebalance_3", scores[selected], 1.0, 3, costs),
            ("rebalance_10", scores[selected], 1.0, 10, costs),
        ]
        for name, test_score, cap_scale, rebalance, test_costs in tests:
            result = run_budgeted(test_score, data, loadings, sleeves, "B03", test_costs,
                                  cap_scale=cap_scale, rebalance_every=rebalance)
            metrics = segment_metrics(result, *VALIDATION)
            kill_rows.append({"test": name, "passed": metrics.get("status") == "COMPLETED"
                              and metrics.get("sharpe", -np.inf) > 0, **metrics})
        for sleeve in sorted(sleeves.unique()):
            reduced = data["eligibility"].copy()
            reduced.loc[:, sleeves.eq(sleeve)] = False
            result = run_budgeted(scores[selected], {**data, "eligibility": reduced}, loadings,
                                  sleeves, "B03", costs)
            metrics = segment_metrics(result, *VALIDATION)
            kill_rows.append({"test": f"remove_sleeve_{sleeve}",
                              "passed": metrics.get("status") == "COMPLETED"
                              and metrics.get("total_return", -np.inf) > 0, **metrics})
        kill_table = pd.DataFrame(kill_rows)
        kill_table.to_csv(output / "kill_tests.csv", index=False)
        kill_passed = bool(kill_table.passed.astype(bool).all())
        kill_lock = {"status": "PASSED" if kill_passed else "REJECTED", "candidate": selected,
                     "kill_tests_sha256": sha256(output / "kill_tests.csv")}
    else:
        pd.DataFrame(columns=["test", "passed", "status"]).to_csv(
            output / "kill_tests.csv", index=False
        )
        kill_lock = {"status": "NOT_RUN_NO_SELECTED_CANDIDATE"}
    (output / "kill_test_lock.json").write_text(json.dumps(kill_lock, indent=2) + "\n")

    test_metrics, test_passed = None, False
    if selected and kill_lock["status"] == "PASSED":
        full = load_data(source, TEST[1])
        full_score = build_time_series_trend(full) * artifacts["time_series_trend"]["slope"]
        full_result = run_budgeted(full_score, full, loadings, sleeves, "B03", costs)
        full_result.daily.to_csv(output / "T01_full_daily.csv")
        test_metrics = segment_metrics(full_result, *TEST)
        test_passed = bool(
            test_metrics.get("status") == "COMPLETED" and test_metrics["sharpe"] > .50
            and test_metrics["cagr"] > .05 and test_metrics["max_drawdown"] >= -.15
        )
        (output / "locked_test.json").write_text(json.dumps(
            {"candidate": selected, "passed": test_passed, "metrics": test_metrics}, indent=2
        ) + "\n")
    summary = {
        "status": "HISTORICAL_ACCEPTED_PENDING_ALPACA_REPLICATION_AND_PROSPECTIVE"
        if test_passed else "REJECTED",
        "selection": selection_lock, "kill_tests": kill_lock, "locked_test": test_metrics,
        "orders_allowed": False,
    }
    (output / "SUMMARY.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path,
                        default=Path("output/cross_asset_etfs_v12/etf_daily.csv"))
    parser.add_argument("--output", type=Path, default=Path("reports/cross_asset_v15"))
    args = parser.parse_args()
    main(args.source, args.output)

"""Evaluate a train-frozen stress scaler on the v13 B03 cross-asset trend portfolio."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd

from portfolio.optimizer import PortfolioCosts
from scripts.evaluate_cross_asset_v12 import (
    TEST,
    TRAIN,
    VALIDATION,
    build_family_scores,
    calibrate,
    load_data,
)
from scripts.evaluate_cross_asset_v13 import (
    asset_sleeves,
    normalized_factor_loadings,
    passes,
    run_budgeted,
    segment_metrics,
)
from scripts.evaluate_equity_v7 import sha256

PROTOCOL = Path("docs/MULTI_ASSET_PROTOCOL_V14.md")


def stress_inputs(data: dict) -> tuple[pd.Series, pd.Series]:
    spy = data["close"]["SPY"]
    drawdown = spy.div(spy.rolling(252, min_periods=252).max()).sub(1).shift(1)
    realized_volatility = data["returns"]["SPY"].rolling(20, min_periods=20).std().mul(
        np.sqrt(252)
    ).shift(1)
    return drawdown, realized_volatility


def build_scaler(data: dict, *, drawdown_levels=(-.10, -.20), volatility_quantiles=(.90, .975)):
    drawdown, volatility = stress_inputs(data)
    training_volatility = volatility.loc[TRAIN[0] : TRAIN[1]].dropna()
    thresholds = training_volatility.quantile(list(volatility_quantiles)).to_numpy(dtype=float)
    scaler = pd.Series(1.0, index=data["returns"].index)
    moderate = drawdown.le(drawdown_levels[0]) | volatility.ge(thresholds[0])
    extreme = drawdown.le(drawdown_levels[1]) | volatility.ge(thresholds[1])
    scaler.loc[moderate.fillna(False)] = .75
    scaler.loc[extreme.fillna(False)] = .50
    artifact = {
        "drawdown_levels": list(drawdown_levels),
        "volatility_quantiles": list(volatility_quantiles),
        "volatility_thresholds": thresholds.tolist(),
        "train_scaler_distribution": scaler.loc[TRAIN[0] : TRAIN[1]].value_counts().sort_index().to_dict(),
    }
    return scaler, artifact


def main(source: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    data = load_data(source, VALIDATION[1])
    trend = build_family_scores(data)["trend"]
    calibration = calibrate(trend, data["returns"])
    if calibration["status"] != "ADMITTED":
        summary = {"status": "REJECTED", "reason": calibration["reason"],
                   "orders_allowed": False, "locked_test_evaluated": False}
        (output / "SUMMARY.json").write_text(json.dumps(summary, indent=2) + "\n")
        return
    score = trend * calibration["slope"]
    loadings = normalized_factor_loadings()
    sleeves = asset_sleeves(loadings.index)
    scaler, scaler_artifact = build_scaler(data)
    pd.DataFrame({"risk_scaler": scaler}).to_csv(output / "risk_scaler.csv")
    costs = PortfolioCosts(
        commission_bps=0, half_spread_bps=1, slippage_bps=1,
        impact_coefficient=.10, borrow_annual=.01, calendar_days=7,
    )
    freeze = {
        "protocol_sha256": sha256(PROTOCOL), "source_sha256": sha256(source),
        "alpha": "frozen_v12_trend", "portfolio": "frozen_v13_B03",
        "calibration": calibration, "scaler": scaler_artifact, "costs": asdict(costs),
        "development_reused": True, "locked_test_loaded_for_selection": False,
    }
    (output / "frozen_run_inputs.json").write_text(json.dumps(freeze, indent=2) + "\n")

    rows, results = [], {}
    for candidate, candidate_scaler, selectable in (
        ("R00", None, False), ("R01", scaler, True),
    ):
        result = run_budgeted(score, data, loadings, sleeves, "B03", costs,
                              risk_scaler=candidate_scaler)
        results[candidate] = result
        result.daily.to_csv(output / f"{candidate}_daily.csv")
        result.rebalances.to_csv(output / f"{candidate}_rebalances.csv")
        train, development = segment_metrics(result, *TRAIN), segment_metrics(result, *VALIDATION)
        eligible = selectable and passes(train, development, budgeted=True)
        row = {"experiment": candidate, "selectable": selectable, "eligible": eligible,
               "status": result.status, "reason": result.reason}
        row.update({f"train_{key}": value for key, value in train.items()})
        row.update({f"development_{key}": value for key, value in development.items()})
        rows.append(row)
    selection = pd.DataFrame(rows).sort_values(["eligible", "experiment"], ascending=[False, True])
    selection.to_csv(output / "candidate_selection.csv", index=False)
    selected = "R01" if bool(selection.set_index("experiment").loc["R01", "eligible"]) else None
    selection_lock = {
        "status": "SELECTED_PENDING_KILL_TESTS" if selected else "REJECTED",
        "reason": None if selected else "STRESS_SCALER_FAILED_FROZEN_GATES",
        "candidate": selected, "locked_test_evaluated": False,
        "selection_sha256": sha256(output / "candidate_selection.csv"),
    }
    (output / "selection_lock.json").write_text(json.dumps(selection_lock, indent=2) + "\n")

    kill_rows = []
    if selected:
        tight_scaler, _ = build_scaler(
            data, drawdown_levels=(-.08, -.16), volatility_quantiles=(.85, .95)
        )
        loose_scaler, _ = build_scaler(
            data, drawdown_levels=(-.12, -.24), volatility_quantiles=(.95, .99)
        )
        tests = [
            ("double_cost", score, scaler, 5, replace(costs, multiplier=2.0)),
            ("signal_delay_1", score.shift(1), scaler, 5, costs),
            ("stress_thresholds_tight", score, tight_scaler, 5, costs),
            ("stress_thresholds_loose", score, loose_scaler, 5, costs),
            ("rebalance_3", score, scaler, 3, costs),
            ("rebalance_10", score, scaler, 10, costs),
        ]
        for name, test_score, test_scaler, rebalance, test_costs in tests:
            result = run_budgeted(test_score, data, loadings, sleeves, "B03", test_costs,
                                  rebalance_every=rebalance, risk_scaler=test_scaler)
            metrics = segment_metrics(result, *VALIDATION)
            kill_rows.append({"test": name, "passed": metrics.get("status") == "COMPLETED"
                              and metrics.get("sharpe", -np.inf) > 0, **metrics})
        for sleeve in sorted(sleeves.unique()):
            reduced = data["eligibility"].copy()
            reduced.loc[:, sleeves.eq(sleeve)] = False
            result = run_budgeted(score, {**data, "eligibility": reduced}, loadings, sleeves,
                                  "B03", costs, risk_scaler=scaler)
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
        full_score = build_family_scores(full)["trend"] * calibration["slope"]
        full_scaler, _ = build_scaler(full)
        full_result = run_budgeted(full_score, full, loadings, sleeves, "B03", costs,
                                   risk_scaler=full_scaler)
        full_result.daily.to_csv(output / "R01_full_daily.csv")
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
    parser.add_argument("--output", type=Path, default=Path("reports/cross_asset_v14"))
    args = parser.parse_args()
    main(args.source, args.output)

"""Train-select a blend of cross-sectional and time-series ETF trend."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd

from portfolio.optimizer import PortfolioCosts
from scripts.evaluate_cross_asset_v12 import TEST, TRAIN, VALIDATION, build_family_scores, calibrate, load_data
from scripts.evaluate_cross_asset_v13 import (
    asset_sleeves,
    normalized_factor_loadings,
    passes,
    run_budgeted,
    segment_metrics,
)
from scripts.evaluate_cross_asset_v15 import build_time_series_trend, calibrate_absolute
from scripts.evaluate_equity_v7 import sha256

PROTOCOL = Path("docs/MULTI_ASSET_PROTOCOL_V16.md")
WEIGHTS = {f"E{int(weight * 100):03d}": weight for weight in (0, .25, .50, .75, 1.0)}


def scores_and_artifacts(data):
    cross = build_family_scores(data)["trend"]
    time = build_time_series_trend(data)
    artifacts = {
        "cross_sectional_trend": calibrate(cross, data["returns"]),
        "time_series_trend": calibrate_absolute(time, data["returns"]),
    }
    scores = {
        "cross": cross * artifacts["cross_sectional_trend"]["slope"],
        "time": time * artifacts["time_series_trend"]["slope"],
    }
    return scores, artifacts


def blend(scores, cross_weight):
    return cross_weight * scores["cross"] + (1 - cross_weight) * scores["time"]


def train_pass(metrics):
    return bool(
        metrics.get("status") == "COMPLETED" and metrics["sharpe"] > .70
        and metrics["cagr"] > .05 and metrics["max_drawdown"] >= -.15
        and metrics["annual_turnover"] <= 25
    )


def main(source: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    train_data = load_data(source, TRAIN[1])
    train_scores, artifacts = scores_and_artifacts(train_data)
    if any(value["status"] != "ADMITTED" for value in artifacts.values()):
        summary = {"status": "REJECTED", "reason": "ENSEMBLE_FAMILY_BLOCKED",
                   "orders_allowed": False, "locked_test_evaluated": False}
        (output / "SUMMARY.json").write_text(json.dumps(summary, indent=2) + "\n")
        return
    loadings = normalized_factor_loadings()
    sleeves = asset_sleeves(loadings.index)
    costs = PortfolioCosts(
        commission_bps=0, half_spread_bps=1, slippage_bps=1,
        impact_coefficient=.10, borrow_annual=.01, calendar_days=7,
    )
    freeze = {
        "protocol_sha256": sha256(PROTOCOL), "source_sha256": sha256(source),
        "artifacts": artifacts, "weights": WEIGHTS, "portfolio": "frozen_v13_B03",
        "costs": asdict(costs), "development_reused": True,
        "locked_test_loaded_for_selection": False,
    }
    (output / "frozen_run_inputs.json").write_text(json.dumps(freeze, indent=2) + "\n")

    rows = []
    for experiment, weight in WEIGHTS.items():
        result = run_budgeted(blend(train_scores, weight), train_data, loadings, sleeves,
                              "B03", costs)
        metrics = segment_metrics(result, *TRAIN)
        row = {"experiment": experiment, "cross_sectional_weight": weight,
               "train_qualified": train_pass(metrics), "status": result.status,
               "reason": result.reason, **{f"train_{key}": value for key, value in metrics.items()}}
        rows.append(row)
    train_table = pd.DataFrame(rows).sort_values(
        ["train_qualified", "train_sharpe", "train_annual_turnover", "experiment"],
        ascending=[False, False, True, True], na_position="last",
    )
    train_table.to_csv(output / "train_selection.csv", index=False)
    qualified = train_table.loc[train_table.train_qualified]
    if qualified.empty:
        selected_experiment = None
    else:
        selected_experiment = str(qualified.iloc[0].experiment)

    development_metrics = None
    selected_weight = WEIGHTS[selected_experiment] if selected_experiment else None
    if selected_experiment:
        development_data = load_data(source, VALIDATION[1])
        development_scores, development_artifacts = scores_and_artifacts(development_data)
        if any(
            not np.isclose(development_artifacts[key]["slope"], artifacts[key]["slope"])
            for key in artifacts
        ):
            raise RuntimeError("train calibration changed when later rows were appended")
        selected_score = blend(development_scores, selected_weight)
        result = run_budgeted(selected_score, development_data, loadings, sleeves, "B03", costs)
        result.daily.to_csv(output / "selected_daily.csv")
        result.rebalances.to_csv(output / "selected_rebalances.csv")
        development_metrics = segment_metrics(result, *VALIDATION)
        eligible = passes(
            segment_metrics(result, *TRAIN), development_metrics, budgeted=True
        )
    else:
        development_data = development_scores = selected_score = None
        eligible = False
    audit = {
        "experiment": selected_experiment, "cross_sectional_weight": selected_weight,
        "eligible": eligible,
        **({f"development_{key}": value for key, value in development_metrics.items()}
           if development_metrics else {}),
    }
    pd.DataFrame([audit]).to_csv(output / "development_audit.csv", index=False)
    selection_lock = {
        "status": "SELECTED_PENDING_KILL_TESTS" if eligible else "REJECTED",
        "reason": None if eligible else (
            "NO_TRAIN_QUALIFIED_BLEND" if selected_experiment is None
            else "TRAIN_SELECTED_BLEND_FAILED_DEVELOPMENT_GATES"
        ),
        "candidate": selected_experiment if eligible else None,
        "train_selected_candidate": selected_experiment,
        "locked_test_evaluated": False,
        "train_selection_sha256": sha256(output / "train_selection.csv"),
        "development_audit_sha256": sha256(output / "development_audit.csv"),
    }
    (output / "selection_lock.json").write_text(json.dumps(selection_lock, indent=2) + "\n")

    kill_rows = []
    if eligible:
        tests = [
            ("double_cost", selected_score, selected_weight, 1.0, 5, replace(costs, multiplier=2)),
            ("signal_delay_1", selected_score.shift(1), selected_weight, 1.0, 5, costs),
            ("blend_minus_25", blend(development_scores, max(0, selected_weight - .25)),
             max(0, selected_weight - .25), 1.0, 5, costs),
            ("blend_plus_25", blend(development_scores, min(1, selected_weight + .25)),
             min(1, selected_weight + .25), 1.0, 5, costs),
            ("caps_tighten_20pct", selected_score, selected_weight, .8, 5, costs),
            ("caps_loosen_20pct", selected_score, selected_weight, 1.2, 5, costs),
            ("rebalance_3", selected_score, selected_weight, 1.0, 3, costs),
            ("rebalance_10", selected_score, selected_weight, 1.0, 10, costs),
        ]
        for name, test_score, _weight, cap_scale, rebalance, test_costs in tests:
            result = run_budgeted(test_score, development_data, loadings, sleeves, "B03",
                                  test_costs, cap_scale=cap_scale, rebalance_every=rebalance)
            metrics = segment_metrics(result, *VALIDATION)
            kill_rows.append({"test": name, "passed": metrics.get("status") == "COMPLETED"
                              and metrics.get("sharpe", -np.inf) > 0, **metrics})
        for sleeve in sorted(sleeves.unique()):
            reduced = development_data["eligibility"].copy()
            reduced.loc[:, sleeves.eq(sleeve)] = False
            result = run_budgeted(selected_score, {**development_data, "eligibility": reduced},
                                  loadings, sleeves, "B03", costs)
            metrics = segment_metrics(result, *VALIDATION)
            kill_rows.append({"test": f"remove_sleeve_{sleeve}",
                              "passed": metrics.get("status") == "COMPLETED"
                              and metrics.get("total_return", -np.inf) > 0, **metrics})
        kill_table = pd.DataFrame(kill_rows)
        kill_table.to_csv(output / "kill_tests.csv", index=False)
        kill_passed = bool(kill_table.passed.astype(bool).all())
        kill_lock = {"status": "PASSED" if kill_passed else "REJECTED",
                     "candidate": selected_experiment,
                     "kill_tests_sha256": sha256(output / "kill_tests.csv")}
    else:
        pd.DataFrame(columns=["test", "passed", "status"]).to_csv(
            output / "kill_tests.csv", index=False
        )
        kill_lock = {"status": "NOT_RUN_NO_SELECTED_CANDIDATE"}
    (output / "kill_test_lock.json").write_text(json.dumps(kill_lock, indent=2) + "\n")

    test_metrics, test_passed = None, False
    if eligible and kill_lock["status"] == "PASSED":
        full = load_data(source, TEST[1])
        full_scores, full_artifacts = scores_and_artifacts(full)
        if any(not np.isclose(full_artifacts[key]["slope"], artifacts[key]["slope"])
               for key in artifacts):
            raise RuntimeError("locked-test append changed train calibration")
        full_score = blend(full_scores, selected_weight)
        full_result = run_budgeted(full_score, full, loadings, sleeves, "B03", costs)
        full_result.daily.to_csv(output / "selected_full_daily.csv")
        test_metrics = segment_metrics(full_result, *TEST)
        test_passed = bool(
            test_metrics.get("status") == "COMPLETED" and test_metrics["sharpe"] > .50
            and test_metrics["cagr"] > .05 and test_metrics["max_drawdown"] >= -.15
        )
        (output / "locked_test.json").write_text(json.dumps(
            {"candidate": selected_experiment, "passed": test_passed,
             "metrics": test_metrics}, indent=2
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
    parser.add_argument("--output", type=Path, default=Path("reports/cross_asset_v16"))
    args = parser.parse_args()
    main(args.source, args.output)

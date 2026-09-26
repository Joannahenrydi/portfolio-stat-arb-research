"""Evaluate frozen trend blends on the expanded 45-ETF v17 universe."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd

from backtest.cross_asset_budget import RiskBudgetConfig, run_risk_budget_backtest
from portfolio.optimizer import PortfolioCosts
from scripts.cross_asset_v17_universe import UNIVERSE, asset_sleeves, factor_loadings
from scripts.evaluate_cross_asset_v12 import TEST, TRAIN, VALIDATION, build_family_scores, calibrate, load_data
from scripts.evaluate_cross_asset_v13 import passes, segment_metrics
from scripts.evaluate_cross_asset_v15 import build_time_series_trend, calibrate_absolute
from scripts.evaluate_cross_asset_v16 import WEIGHTS, blend, train_pass
from scripts.evaluate_equity_v7 import sha256

PROTOCOL = Path("docs/MULTI_ASSET_PROTOCOL_V17.md")
FACTOR_CAPS = pd.Series(
    {"equity": .50, "duration": .50, "credit": .30, "commodity": .50, "usd": .40}
)
SLEEVE_CAPS = pd.Series(
    {"equity": .60, "rates": .60, "credit": .40, "metals": .40, "commodity": .60, "usd": .30}
)
CONFIG = RiskBudgetConfig(max_name=.10, net_cap=.60)


def normalized_loadings():
    frame = factor_loadings()
    return frame.div(frame.abs().max().replace(0, 1), axis=1)


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


def run_one(score, data, loadings, sleeves, costs, *, cap_scale=1.0, rebalance_every=5,
            volatility_cap=.10):
    return run_risk_budget_backtest(
        score, data["returns"], data["adv"], data["eligibility"] & score.notna(),
        loadings, FACTOR_CAPS * cap_scale, sleeves, SLEEVE_CAPS * cap_scale,
        config=replace(
            CONFIG, rebalance_every=rebalance_every, net_cap=min(CONFIG.net_cap * cap_scale, 1.0),
            annual_volatility_cap=volatility_cap,
        ),
        costs=costs,
    )


def main(source: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    train_data = load_data(source, TRAIN[1], UNIVERSE)
    train_scores, artifacts = scores_and_artifacts(train_data)
    if any(value["status"] != "ADMITTED" for value in artifacts.values()):
        summary = {"status": "REJECTED", "reason": "EXPANDED_ENSEMBLE_FAMILY_BLOCKED",
                   "orders_allowed": False, "locked_test_evaluated": False}
        (output / "SUMMARY.json").write_text(json.dumps(summary, indent=2) + "\n")
        return
    loadings = normalized_loadings()
    sleeves = asset_sleeves()
    costs = PortfolioCosts(
        commission_bps=0, half_spread_bps=1, slippage_bps=1,
        impact_coefficient=.10, borrow_annual=.01, calendar_days=7,
    )
    freeze = {
        "protocol_sha256": sha256(PROTOCOL), "source_sha256": sha256(source),
        "universe": UNIVERSE, "artifacts": artifacts, "weights": WEIGHTS,
        "factor_caps": FACTOR_CAPS.to_dict(), "sleeve_caps": SLEEVE_CAPS.to_dict(),
        "config": asdict(CONFIG), "costs": asdict(costs), "development_reused": True,
        "locked_test_loaded_for_selection": False,
    }
    (output / "frozen_run_inputs.json").write_text(json.dumps(freeze, indent=2) + "\n")
    loadings.to_csv(output / "normalized_factor_loadings.csv")
    sleeves.to_csv(output / "asset_sleeves.csv")

    rows = []
    for experiment, weight in WEIGHTS.items():
        result = run_one(blend(train_scores, weight), train_data, loadings, sleeves, costs)
        metrics = segment_metrics(result, *TRAIN)
        rows.append({
            "experiment": experiment, "cross_sectional_weight": weight,
            "train_qualified": train_pass(metrics), "status": result.status,
            "reason": result.reason, **{f"train_{key}": value for key, value in metrics.items()},
        })
    train_table = pd.DataFrame(rows).sort_values(
        ["train_qualified", "train_sharpe", "train_annual_turnover", "experiment"],
        ascending=[False, False, True, True], na_position="last",
    )
    train_table.to_csv(output / "train_selection.csv", index=False)
    qualified = train_table.loc[train_table.train_qualified]
    selected_experiment = str(qualified.iloc[0].experiment) if not qualified.empty else None
    selected_weight = WEIGHTS[selected_experiment] if selected_experiment else None

    development_metrics = None
    if selected_experiment:
        development_data = load_data(source, VALIDATION[1], UNIVERSE)
        development_scores, development_artifacts = scores_and_artifacts(development_data)
        if any(not np.isclose(development_artifacts[key]["slope"], artifacts[key]["slope"])
               for key in artifacts):
            raise RuntimeError("appending development rows changed train calibration")
        selected_score = blend(development_scores, selected_weight)
        result = run_one(selected_score, development_data, loadings, sleeves, costs)
        result.daily.to_csv(output / "selected_daily.csv")
        result.rebalances.to_csv(output / "selected_rebalances.csv")
        train_metrics = segment_metrics(result, *TRAIN)
        development_metrics = segment_metrics(result, *VALIDATION)
        eligible = passes(train_metrics, development_metrics, budgeted=True)
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
            else "EXPANDED_TRAIN_WINNER_FAILED_DEVELOPMENT_GATES"
        ),
        "candidate": selected_experiment if eligible else None,
        "train_selected_candidate": selected_experiment, "locked_test_evaluated": False,
        "train_selection_sha256": sha256(output / "train_selection.csv"),
        "development_audit_sha256": sha256(output / "development_audit.csv"),
    }
    (output / "selection_lock.json").write_text(json.dumps(selection_lock, indent=2) + "\n")

    if eligible:
        kill_rows = []
        tests = [
            ("double_cost", selected_score, 1.0, 5, replace(costs, multiplier=2)),
            ("signal_delay_1", selected_score.shift(1), 1.0, 5, costs),
            ("blend_minus_25", blend(development_scores, max(0, selected_weight - .25)), 1.0, 5, costs),
            ("blend_plus_25", blend(development_scores, min(1, selected_weight + .25)), 1.0, 5, costs),
            ("caps_tighten_20pct", selected_score, .8, 5, costs),
            ("caps_loosen_20pct", selected_score, 1.2, 5, costs),
            ("rebalance_3", selected_score, 1.0, 3, costs),
            ("rebalance_10", selected_score, 1.0, 10, costs),
        ]
        for name, test_score, cap_scale, rebalance, test_costs in tests:
            result = run_one(test_score, development_data, loadings, sleeves, test_costs,
                             cap_scale=cap_scale, rebalance_every=rebalance)
            metrics = segment_metrics(result, *VALIDATION)
            kill_rows.append({"test": name, "passed": metrics.get("status") == "COMPLETED"
                              and metrics.get("sharpe", -np.inf) > 0, **metrics})
        for sleeve in sorted(sleeves.unique()):
            reduced = development_data["eligibility"].copy()
            reduced.loc[:, sleeves.eq(sleeve)] = False
            result = run_one(selected_score, {**development_data, "eligibility": reduced},
                             loadings, sleeves, costs)
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
        full = load_data(source, TEST[1], UNIVERSE)
        full_scores, full_artifacts = scores_and_artifacts(full)
        if any(not np.isclose(full_artifacts[key]["slope"], artifacts[key]["slope"])
               for key in artifacts):
            raise RuntimeError("locked-test append changed train calibration")
        full_result = run_one(blend(full_scores, selected_weight), full, loadings, sleeves, costs)
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
                        default=Path("output/cross_asset_etfs_v17/etf_daily.csv"))
    parser.add_argument("--output", type=Path, default=Path("reports/cross_asset_v17"))
    args = parser.parse_args()
    main(args.source, args.output)

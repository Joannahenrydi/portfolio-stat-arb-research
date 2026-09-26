"""Test whether macro risk budgets preserve the frozen v12 ETF trend alpha."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd

from backtest.cross_asset import CrossAssetConfig, run_cross_asset_backtest
from backtest.cross_asset_budget import RiskBudgetConfig, run_risk_budget_backtest
from backtest.engine import performance_metrics
from portfolio.optimizer import PortfolioCosts
from scripts.evaluate_cross_asset_v12 import (
    TEST,
    TRAIN,
    VALIDATION,
    build_family_scores,
    calibrate,
    factor_loadings,
    load_data,
)
from scripts.evaluate_equity_v7 import sha256

PROTOCOL = Path("docs/MULTI_ASSET_PROTOCOL_V13.md")
SLEEVES = {
    "SPY": "equity", "QQQ": "equity", "IWM": "equity", "EFA": "equity",
    "EEM": "equity", "VNQ": "equity", "SHY": "rates", "IEF": "rates",
    "TLT": "rates", "TIP": "rates", "LQD": "credit", "HYG": "credit",
    "GLD": "metals", "SLV": "metals", "DBC": "commodity", "USO": "commodity",
    "UNG": "commodity", "DBA": "commodity", "UUP": "usd",
}
FACTOR_CAPS = {
    "B01": {"equity": .10, "duration": .10, "credit": .05, "commodity": .10, "usd": .08},
    "B02": {"equity": .25, "duration": .25, "credit": .15, "commodity": .25, "usd": .20},
    "B03": {"equity": .50, "duration": .50, "credit": .30, "commodity": .50, "usd": .40},
}
SLEEVE_CAPS = {
    "B01": {"equity": .20, "rates": .20, "credit": .15, "metals": .15,
            "commodity": .20, "usd": .10},
    "B02": {"equity": .40, "rates": .40, "credit": .30, "metals": .30,
            "commodity": .40, "usd": .15},
    "B03": {"equity": .60, "rates": .60, "credit": .40, "metals": .40,
            "commodity": .60, "usd": .15},
}
NET_CAPS = {"B01": .10, "B02": .35, "B03": .60}


def normalized_factor_loadings() -> pd.DataFrame:
    """Normalize heterogeneous loading units without changing their signs or zeros."""
    frame = factor_loadings()
    scale = frame.abs().max().replace(0, 1)
    return frame.div(scale, axis=1)


def asset_sleeves(index: pd.Index) -> pd.Series:
    sleeves = pd.Series(SLEEVES, name="sleeve").reindex(index)
    if sleeves.isna().any():
        raise ValueError(f"missing sleeve labels: {sleeves.index[sleeves.isna()].tolist()}")
    return sleeves


def segment_metrics(result, start, end) -> dict:
    if result.status != "COMPLETED":
        return {"status": result.status, "reason": result.reason}
    daily = result.daily.loc[start:end]
    if daily.empty:
        return {"status": "INVALID", "reason": "EMPTY_SEGMENT"}
    net = performance_metrics(daily.net_return)
    gross = performance_metrics(daily.gross_return)
    years = len(daily) / 252
    rebalances = result.rebalances.loc[start:end]
    yearly = (1 + daily.net_return).groupby(daily.index.year).prod() - 1
    net.update(
        status="COMPLETED", gross_sharpe=gross["sharpe"], gross_cagr=gross["cagr"],
        annual_turnover=float(daily.turnover.sum() / years),
        annual_transaction_cost=float(daily.transaction_cost.sum() / years),
        annual_borrow_cost=float(daily.borrow_cost.sum() / years),
        maximum_abs_net=float(rebalances.net.abs().max()),
        maximum_net_budget_ratio=float(rebalances.get("net_budget_ratio", pd.Series(0.0)).max()),
        maximum_factor_budget_ratio=float(
            rebalances.get("maximum_factor_budget_ratio", pd.Series(0.0)).max()
        ),
        maximum_sleeve_budget_ratio=float(
            rebalances.get("maximum_sleeve_budget_ratio", pd.Series(0.0)).max()
        ),
        average_gross=float(daily.gross.mean()),
        average_eligible_assets=float(rebalances.eligible_assets.mean()),
        worst_calendar_year_return=float(yearly.min()),
    )
    return net


def passes(train: dict, validation: dict, *, budgeted: bool) -> bool:
    if train.get("status") != "COMPLETED" or validation.get("status") != "COMPLETED":
        return False
    constraint_ok = True
    if budgeted:
        constraint_ok = all(
            validation[key] <= 1.0001 for key in (
                "maximum_net_budget_ratio", "maximum_factor_budget_ratio",
                "maximum_sleeve_budget_ratio",
            )
        )
    return bool(
        train["sharpe"] > .70 and validation["sharpe"] > .50
        and validation["cagr"] > .05
        and train["max_drawdown"] >= -.15 and validation["max_drawdown"] >= -.15
        and validation["annual_turnover"] <= 25 and constraint_ok
    )


def run_budgeted(score, data, loadings, sleeves, candidate, costs, *, cap_scale=1.0,
                 rebalance_every=5, risk_scaler=None):
    factor_caps = pd.Series(FACTOR_CAPS[candidate], dtype=float) * cap_scale
    sleeve_caps = pd.Series(SLEEVE_CAPS[candidate], dtype=float) * cap_scale
    config = RiskBudgetConfig(
        rebalance_every=rebalance_every,
        net_cap=min(NET_CAPS[candidate] * cap_scale, 1.0),
    )
    return run_risk_budget_backtest(
        score, data["returns"], data["adv"], data["eligibility"] & score.notna(),
        loadings, factor_caps, sleeves, sleeve_caps, config=config, costs=costs,
        risk_scaler=risk_scaler,
    )


def main(source: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    data = load_data(source, VALIDATION[1])
    trend = build_family_scores(data)["trend"]
    artifact = calibrate(trend, data["returns"])
    pd.DataFrame([{"family": "trend", **artifact}]).to_csv(
        output / "train_family_calibration.csv", index=False
    )
    if artifact["status"] != "ADMITTED":
        summary = {"status": "REJECTED", "reason": artifact["reason"],
                   "orders_allowed": False, "locked_test_evaluated": False}
        (output / "SUMMARY.json").write_text(json.dumps(summary, indent=2) + "\n")
        return
    score = trend * artifact["slope"]
    raw_loadings = factor_loadings()
    loadings = normalized_factor_loadings()
    sleeves = asset_sleeves(loadings.index)
    loadings.to_csv(output / "normalized_factor_loadings.csv")
    sleeves.to_csv(output / "asset_sleeves.csv")
    costs = PortfolioCosts(
        commission_bps=0, half_spread_bps=1, slippage_bps=1,
        impact_coefficient=.10, borrow_annual=.01, calendar_days=7,
    )
    freeze = {
        "protocol_sha256": sha256(PROTOCOL), "source_sha256": sha256(source),
        "alpha": "frozen_v12_trend", "calibration": artifact,
        "costs": asdict(costs), "factor_caps": FACTOR_CAPS,
        "sleeve_caps": SLEEVE_CAPS, "net_caps": NET_CAPS,
        "locked_test_loaded_for_selection": False,
    }
    (output / "frozen_run_inputs.json").write_text(json.dumps(freeze, indent=2) + "\n")

    rows, runs = [], {}
    exact = run_cross_asset_backtest(
        score, data["returns"], data["adv"], data["eligibility"] & score.notna(),
        raw_loadings, config=CrossAssetConfig(rebalance_every=5), costs=costs,
    )
    runs["B00"] = exact
    exact.daily.to_csv(output / "B00_daily.csv")
    exact.rebalances.to_csv(output / "B00_rebalances.csv")
    train, validation = segment_metrics(exact, *TRAIN), segment_metrics(exact, *VALIDATION)
    row = {"experiment": "B00", "constraint_regime": "exact", "selectable": False,
           "eligible": False, "status": exact.status, "reason": exact.reason}
    row.update({f"train_{key}": value for key, value in train.items()})
    row.update({f"validation_{key}": value for key, value in validation.items()})
    rows.append(row)

    for candidate, regime in (("B01", "tight"), ("B02", "medium"), ("B03", "loose")):
        result = run_budgeted(score, data, loadings, sleeves, candidate, costs)
        runs[candidate] = result
        result.daily.to_csv(output / f"{candidate}_daily.csv")
        result.rebalances.to_csv(output / f"{candidate}_rebalances.csv")
        train, validation = segment_metrics(result, *TRAIN), segment_metrics(result, *VALIDATION)
        eligible = passes(train, validation, budgeted=True)
        row = {"experiment": candidate, "constraint_regime": regime, "selectable": True,
               "eligible": eligible, "status": result.status, "reason": result.reason}
        row.update({f"train_{key}": value for key, value in train.items()})
        row.update({f"validation_{key}": value for key, value in validation.items()})
        row["robust_sharpe"] = min(train.get("sharpe", -np.inf), validation.get("sharpe", -np.inf))
        rows.append(row)
    selection = pd.DataFrame(rows).sort_values(
        ["eligible", "robust_sharpe", "validation_cagr", "experiment"],
        ascending=[False, False, False, True], na_position="last",
    )
    selection.to_csv(output / "candidate_selection.csv", index=False)
    selectable = selection.loc[selection.selectable & selection.eligible]
    selected = str(selectable.iloc[0].experiment) if not selectable.empty else None
    selection_lock = {
        "status": "SELECTED_PENDING_KILL_TESTS" if selected else "REJECTED",
        "reason": None if selected else "NO_V13_RISK_BUDGET_PASSED_TRAIN_VALIDATION_GATES",
        "candidate": selected, "locked_test_evaluated": False,
        "selection_sha256": sha256(output / "candidate_selection.csv"),
    }
    (output / "selection_lock.json").write_text(json.dumps(selection_lock, indent=2) + "\n")

    kill_rows = []
    if selected:
        tests = [
            ("double_cost", score, 1.0, 5, replace(costs, multiplier=2.0)),
            ("signal_delay_1", score.shift(1), 1.0, 5, costs),
            ("caps_tighten_20pct", score, .8, 5, costs),
            ("caps_loosen_20pct", score, 1.2, 5, costs),
            ("rebalance_3", score, 1.0, 3, costs),
            ("rebalance_10", score, 1.0, 10, costs),
        ]
        for name, test_score, scale, rebalance, test_costs in tests:
            result = run_budgeted(test_score, data, loadings, sleeves, selected, test_costs,
                                  cap_scale=scale, rebalance_every=rebalance)
            metrics = segment_metrics(result, *VALIDATION)
            kill_rows.append({"test": name, "passed": metrics.get("status") == "COMPLETED"
                              and metrics.get("sharpe", -np.inf) > 0, **metrics})
        for sleeve in sorted(sleeves.unique()):
            reduced = data["eligibility"].copy()
            reduced.loc[:, sleeves.eq(sleeve)] = False
            reduced_data = {**data, "eligibility": reduced}
            result = run_budgeted(score, reduced_data, loadings, sleeves, selected, costs)
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
        full_trend = build_family_scores(full)["trend"]
        full_score = full_trend * artifact["slope"]
        full_result = run_budgeted(full_score, full, loadings, sleeves, selected, costs)
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
    parser.add_argument("--output", type=Path, default=Path("reports/cross_asset_v13"))
    args = parser.parse_args()
    main(args.source, args.output)

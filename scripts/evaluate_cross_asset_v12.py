"""Evaluate the frozen ETF cross-asset relative-value strategy."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd

from backtest.cross_asset import CrossAssetConfig, CrossAssetResult, run_cross_asset_backtest
from backtest.engine import performance_metrics
from features.adaptive_alpha import adaptive_kalman_innovation
from features.alphas import forward_total_return_labels
from portfolio.optimizer import PortfolioCosts
from scripts.collect_cross_asset_etfs import UNIVERSE
from scripts.evaluate_equity_v7 import sha256

TRAIN = (pd.Timestamp("2008-01-02"), pd.Timestamp("2016-12-30"))
VALIDATION = (pd.Timestamp("2017-01-03"), pd.Timestamp("2020-12-31"))
TEST = (pd.Timestamp("2021-01-04"), pd.Timestamp("2024-12-31"))
PROTOCOL = Path("docs/MULTI_ASSET_PROTOCOL_V12.md")
PAIRS = [
    ("SPY", "TLT"), ("QQQ", "IWM"), ("HYG", "LQD"), ("HYG", "SPY"),
    ("GLD", "TIP"), ("SLV", "GLD"), ("DBC", "UUP"), ("USO", "DBC"),
    ("LQD", "IEF"),
]


def factor_loadings() -> pd.DataFrame:
    frame = pd.DataFrame(
        0.0, index=UNIVERSE,
        columns=["equity", "duration", "credit", "commodity", "usd"],
    )
    frame.loc[["SPY", "QQQ", "IWM"], "equity"] = [1.0, 1.1, 1.1]
    frame.loc[["EFA", "EEM", "VNQ", "LQD", "HYG"], "equity"] = [0.9, 1.0, 0.7, 0.1, 0.4]
    frame.loc[["SHY", "IEF", "TLT", "TIP", "LQD", "HYG"], "duration"] = [1, 4, 15, 7, 8, 3]
    frame.loc[["LQD", "HYG"], "credit"] = [0.4, 1.0]
    frame.loc[["GLD", "SLV", "DBC", "USO", "UNG", "DBA"], "commodity"] = [0.3, 0.5, 1, 1.2, 1.5, 0.8]
    frame.loc["UUP", "usd"] = 1.0
    frame.loc[["EFA", "EEM"], "usd"] = -0.3
    frame.loc[["GLD", "SLV", "DBC", "USO", "UNG", "DBA"], "usd"] = -0.3
    return frame


def load_data(path: Path, end: pd.Timestamp, universe=UNIVERSE) -> dict:
    raw = pd.read_csv(path, parse_dates=["date"])
    raw = raw.loc[raw.date.le(end) & raw.symbol.isin(universe)].copy()
    dates = pd.DatetimeIndex(sorted(raw.loc[raw.symbol.eq("SPY"), "date"].unique()))

    def pivot(field):
        return raw.pivot(index="date", columns="symbol", values=field).reindex(
            index=dates, columns=universe
        )

    close = pivot("adj_close")
    volume = pivot("volume")
    returns = close.pct_change(fill_method=None)
    adv = (pivot("close") * volume).rolling(60, min_periods=60).median().shift(1)
    history = returns.notna().rolling(252, min_periods=1).sum().shift(1)
    eligibility = returns.notna() & close.notna() & volume.gt(0) & adv.gt(0) & history.ge(252)
    return {"close": close, "volume": volume, "returns": returns, "adv": adv,
            "eligibility": eligibility}


def zscore(frame: pd.DataFrame, eligibility: pd.DataFrame) -> pd.DataFrame:
    usable = frame.where(eligibility)
    centered = usable.sub(usable.mean(axis=1), axis=0)
    scale = usable.std(axis=1, ddof=1).where(usable.count(axis=1).ge(3))
    return centered.div(scale.where(scale.gt(0)), axis=0).clip(-3, 3)


def pair_score(values, index, columns) -> pd.DataFrame:
    total = pd.DataFrame(0.0, index=index, columns=columns)
    count = pd.DataFrame(0.0, index=index, columns=columns)
    for (left, right), signal in values.items():
        valid = signal.notna()
        total.loc[valid, left] += signal.loc[valid]
        total.loc[valid, right] -= signal.loc[valid]
        count.loc[valid, left] += 1
        count.loc[valid, right] += 1
    return total.div(count.where(count.gt(0)))


def build_family_scores(data: dict, pairs=PAIRS) -> dict[str, pd.DataFrame]:
    returns, close, eligibility = data["returns"], data["close"], data["eligibility"]
    log_price = np.log(close)
    volatility = returns.rolling(60, min_periods=60).std().shift(1)
    trend_parts = []
    for horizon in (63, 126, 252):
        raw = log_price.shift(21).sub(log_price.shift(horizon))
        trend_parts.append(raw.div(volatility * np.sqrt(horizon - 21)))
    trend = zscore(sum(trend_parts) / len(trend_parts), eligibility)
    kalman_values, ratio_values = {}, {}
    for left, right in pairs:
        state = adaptive_kalman_innovation(returns[[left]], returns[right], mode="qr_adaptive")
        kalman_values[(left, right)] = state.signal[left]
        ratio = log_price[left] - log_price[right]
        mean = ratio.rolling(252, min_periods=252).mean().shift(1)
        std = ratio.rolling(252, min_periods=252).std().shift(1)
        ratio_values[(left, right)] = -(ratio - mean).div(std.where(std.gt(0))).clip(-3, 3)
    kalman = zscore(pair_score(kalman_values, returns.index, list(returns.columns)), eligibility)
    ratio = zscore(pair_score(ratio_values, returns.index, list(returns.columns)), eligibility)
    return {"trend": trend, "kalman_relative_value": kalman, "ratio_mean_reversion": ratio}


def calibrate(score, returns) -> dict:
    labels, label_end = forward_total_return_labels(returns, 5)
    completed = label_end.le(TRAIN[1])
    centered = labels.sub(labels.mean(axis=1), axis=0)
    x = score.loc[TRAIN[0] : TRAIN[1]].where(completed.loc[TRAIN[0] : TRAIN[1]]).to_numpy().ravel()
    y = centered.loc[TRAIN[0] : TRAIN[1]].where(completed.loc[TRAIN[0] : TRAIN[1]]).to_numpy().ravel()
    usable = np.isfinite(x) & np.isfinite(y)
    if usable.sum() < 1000:
        return {"status": "BLOCKED", "reason": "INSUFFICIENT_CALIBRATION", "slope": np.nan}
    slope = float(x[usable] @ y[usable] / (x[usable] @ x[usable] + 1e-6))
    daily_ic = score.where(completed).corrwith(
        labels.where(completed), axis=1, method="spearman"
    ).loc[TRAIN[0] : TRAIN[1]]
    mean_ic, ic_std = float(daily_ic.mean()), float(daily_ic.std(ddof=1))
    return {
        "status": "ADMITTED" if slope > 0 else "BLOCKED",
        "reason": "OK" if slope > 0 else "NONPOSITIVE_TRAIN_SLOPE",
        "slope": slope, "observations": int(usable.sum()), "mean_rank_ic": mean_ic,
        "icir": mean_ic / ic_std if ic_std > 0 else np.nan,
        "digest": hashlib.sha256(np.c_[x[usable], y[usable]].tobytes()).hexdigest(),
    }


def segment_metrics(result: CrossAssetResult, start, end) -> dict:
    if result.status != "COMPLETED":
        return {"status": result.status, "reason": result.reason}
    daily = result.daily.loc[start:end]
    metrics = performance_metrics(daily.net_return)
    gross = performance_metrics(daily.gross_return)
    years = len(daily) / 252
    rebalances = result.rebalances.loc[start:end]
    yearly = (1 + daily.net_return).groupby(daily.index.year).prod() - 1
    metrics.update(
        status="COMPLETED", gross_sharpe=gross["sharpe"], gross_cagr=gross["cagr"],
        annual_turnover=float(daily.turnover.sum() / years),
        total_transaction_cost=float(daily.transaction_cost.sum()),
        total_borrow_cost=float(daily.borrow_cost.sum()),
        maximum_abs_net=float(rebalances.net.abs().max()),
        maximum_factor_exposure=float(rebalances.max_factor_exposure.abs().max()),
        average_eligible_assets=float(rebalances.eligible_assets.mean()),
        constraint_failures=0, worst_calendar_year_return=float(yearly.min()),
    )
    return metrics


def passes(train, validation) -> bool:
    return bool(
        train.get("status") == validation.get("status") == "COMPLETED"
        and train["sharpe"] > 0.70 and validation["sharpe"] > 0.50
        and validation["cagr"] > 0.05
        and train["max_drawdown"] >= -0.15 and validation["max_drawdown"] >= -0.15
        and validation["annual_turnover"] <= 25
        and validation["maximum_abs_net"] <= 1e-8
        and validation["maximum_factor_exposure"] <= 1e-8
    )


def run_one(score, data, loadings, config, costs):
    return run_cross_asset_backtest(
        score, data["returns"], data["adv"], data["eligibility"] & score.notna(),
        loadings, config=config, costs=costs,
    )


def compose_candidates(families, artifacts):
    admitted = [name for name, artifact in artifacts.items() if artifact["status"] == "ADMITTED"]
    expected = {name: families[name] * artifacts[name]["slope"] for name in admitted}
    blocked, scores, compositions = {}, {}, {}
    singles = {"M00": "trend", "M01": "kalman_relative_value", "M02": "ratio_mean_reversion"}
    for candidate, family in singles.items():
        if family in admitted:
            scores[candidate], compositions[candidate] = expected[family], {family: 1.0}
        else:
            blocked[candidate] = artifacts[family]["reason"]
    if len(admitted) >= 2:
        equal = {name: 1 / len(admitted) for name in admitted}
        scores["M03"] = sum(equal[name] * expected[name] for name in admitted)
        compositions["M03"] = equal
        raw = np.array([max(artifacts[name]["icir"], 0) for name in admitted])
        if raw.sum() > 0:
            weights = {name: value for name, value in zip(admitted, raw / raw.sum())}
            scores["M04"] = sum(weights[name] * expected[name] for name in admitted)
            scores["M05"] = scores["M04"]
            compositions["M04"] = compositions["M05"] = weights
        else:
            blocked["M04"] = blocked["M05"] = "NO_POSITIVE_TRAIN_ICIR"
    else:
        blocked["M03"] = blocked["M04"] = blocked["M05"] = "FEWER_THAN_TWO_ADMITTED_FAMILIES"
    return admitted, scores, compositions, blocked


def main(source: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    data = load_data(source, VALIDATION[1])
    families = build_family_scores(data)
    artifacts = {name: calibrate(score, data["returns"]) for name, score in families.items()}
    pd.DataFrame([{"family": name, **artifact} for name, artifact in artifacts.items()]).to_csv(
        output / "train_family_calibration.csv", index=False
    )
    admitted, scores, compositions, blocked = compose_candidates(families, artifacts)
    loadings = factor_loadings()
    loadings.to_csv(output / "factor_loadings.csv")
    costs = PortfolioCosts(
        commission_bps=0, half_spread_bps=1, slippage_bps=1,
        impact_coefficient=0.10, borrow_annual=0.01, calendar_days=7,
    )
    configs = {
        name: CrossAssetConfig(rebalance_every=10 if name == "M05" else 5)
        for name in ("M00", "M01", "M02", "M03", "M04", "M05")
    }
    freeze = {
        "protocol_sha256": sha256(PROTOCOL), "source_sha256": sha256(source),
        "admitted_families": admitted, "blocked_candidates": blocked,
        "compositions": compositions, "pairs": PAIRS, "costs": asdict(costs),
        "configs": {name: asdict(config) for name, config in configs.items()},
        "locked_test_loaded_for_selection": False,
    }
    (output / "frozen_run_inputs.json").write_text(json.dumps(freeze, indent=2) + "\n")

    runs, rows = {}, []
    for name in configs:
        if name in blocked:
            rows.append({"experiment": name, "status": "BLOCKED", "reason": blocked[name], "eligible": False})
            continue
        result = run_one(scores[name], data, loadings, configs[name], costs)
        runs[name] = result
        train, validation = segment_metrics(result, *TRAIN), segment_metrics(result, *VALIDATION)
        row = {"experiment": name, "status": result.status, "reason": result.reason,
               "eligible": passes(train, validation)}
        row.update({f"train_{key}": value for key, value in train.items()})
        row.update({f"validation_{key}": value for key, value in validation.items()})
        if result.status == "COMPLETED":
            row["robust_sharpe"] = min(train["sharpe"], validation["sharpe"])
        rows.append(row)
    selection = pd.DataFrame(rows).sort_values(
        ["eligible", "robust_sharpe", "validation_cagr", "validation_annual_turnover", "experiment"],
        ascending=[False, False, False, True, True], na_position="last",
    )
    selection.to_csv(output / "candidate_selection.csv", index=False)
    selected = str(selection.loc[selection.eligible].iloc[0].experiment) if selection.eligible.any() else None
    selection_lock = {
        "status": "SELECTED_PENDING_KILL_TESTS" if selected else "REJECTED",
        "reason": None if selected else "NO_V12_CANDIDATE_PASSED_TRAIN_VALIDATION_GATES",
        "candidate": selected, "locked_test_evaluated": False,
        "selection_sha256": sha256(output / "candidate_selection.csv"),
    }
    (output / "selection_lock.json").write_text(json.dumps(selection_lock, indent=2) + "\n")

    kill_rows = []
    if selected is None:
        pd.DataFrame(columns=["test", "passed", "status"]).to_csv(output / "kill_tests.csv", index=False)
        kill_lock = {"status": "NOT_RUN_NO_SELECTED_CANDIDATE"}
    else:
        tests = [
            ("double_cost", scores[selected], configs[selected], replace(costs, multiplier=2.0)),
            ("signal_delay_1", scores[selected].shift(1), configs[selected], costs),
            ("rebalance_3", scores[selected], replace(configs[selected], rebalance_every=3), costs),
            ("rebalance_10", scores[selected], replace(configs[selected], rebalance_every=10), costs),
        ]
        for test_name, score, config, test_costs in tests:
            result = run_one(score, data, loadings, config, test_costs)
            metrics = segment_metrics(result, *VALIDATION)
            passed_test = result.status == "COMPLETED" and metrics["sharpe"] > 0
            kill_rows.append({"test": test_name, "passed": passed_test, **metrics})
        relative = {"kalman_relative_value", "ratio_mean_reversion"} & set(compositions[selected])
        if relative:
            for pair in PAIRS:
                reduced_families = build_family_scores(data, [item for item in PAIRS if item != pair])
                reduced_expected = {
                    name: reduced_families[name] * artifacts[name]["slope"] for name in admitted
                }
                reduced_score = sum(
                    weight * reduced_expected[name]
                    for name, weight in compositions[selected].items()
                )
                result = run_one(reduced_score, data, loadings, configs[selected], costs)
                metrics = segment_metrics(result, *VALIDATION)
                kill_rows.append({"test": f"remove_pair_{pair[0]}_{pair[1]}",
                                  "passed": result.status == "COMPLETED" and metrics["total_return"] > 0,
                                  **metrics})
        else:
            kill_rows.append({"test": "pair_removal", "passed": np.nan, "status": "NOT_APPLICABLE"})
        kill_table = pd.DataFrame(kill_rows)
        kill_table.to_csv(output / "kill_tests.csv", index=False)
        applicable = kill_table.loc[kill_table.status.ne("NOT_APPLICABLE"), "passed"].astype(bool)
        kill_lock = {"status": "PASSED" if applicable.all() else "REJECTED", "candidate": selected,
                     "kill_tests_sha256": sha256(output / "kill_tests.csv")}
    (output / "kill_test_lock.json").write_text(json.dumps(kill_lock, indent=2) + "\n")

    test_metrics, test_passed = None, False
    if selected and kill_lock["status"] == "PASSED":
        full = load_data(source, TEST[1])
        full_families = build_family_scores(full)
        full_expected = {
            name: full_families[name] * artifacts[name]["slope"] for name in admitted
        }
        full_score = sum(
            weight * full_expected[name] for name, weight in compositions[selected].items()
        )
        full_result = run_one(full_score, full, loadings, configs[selected], costs)
        test_metrics = segment_metrics(full_result, *TEST)
        test_passed = (
            test_metrics.get("status") == "COMPLETED" and test_metrics["sharpe"] > 0.50
            and test_metrics["cagr"] > 0.05 and test_metrics["max_drawdown"] >= -0.15
        )
        (output / "locked_test.json").write_text(json.dumps(
            {"candidate": selected, "passed": test_passed, "metrics": test_metrics}, indent=2
        ) + "\n")
    status = (
        "HISTORICAL_ACCEPTED_PENDING_ALPACA_REPLICATION_AND_PROSPECTIVE"
        if test_passed else "REJECTED"
    )
    summary = {"status": status, "selection": selection_lock, "kill_tests": kill_lock,
               "locked_test": test_metrics, "orders_allowed": False}
    (output / "SUMMARY.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("output/cross_asset_etfs_v12/etf_daily.csv"))
    parser.add_argument("--output", type=Path, default=Path("reports/cross_asset_v12"))
    args = parser.parse_args()
    main(args.source, args.output)

"""Evaluate the frozen paper-inspired v8 strategy-sleeve allocator."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd

from backtest.engine import performance_metrics
from backtest.residual_reversal import ResidualReversalConfig, fit_stress_artifact
from portfolio.optimizer import PortfolioCosts
from portfolio.sleeve_allocator import (
    SleeveAllocationResult,
    SleeveAllocatorConfig,
    run_sleeve_allocation,
)
from scripts.evaluate_equity_v2 import AUDIT, TRAIN, VALIDATION
from scripts.evaluate_equity_v7 import load_panels, run_candidate

PROTOCOL = Path("docs/EQUITY_RESEARCH_PROTOCOL_V8.md")
LEGACY = Path("reports/portfolio_2026-09-20")
V7 = Path("reports/equity_v7/R01_daily.csv")
SLEEVES = ("legacy_reversal", "kalman", "dislocation", "legacy_blend", "v7_R01")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_return_file(path: Path, date_column: str = "date") -> pd.Series:
    frame = pd.read_csv(path, usecols=[date_column, "net_return"])
    series = frame.set_index(pd.to_datetime(frame[date_column])).net_return.astype(float)
    series.index.name = "session"
    return series


def _legacy_sleeve(stem: str, include_audit: bool) -> pd.Series:
    pieces = [
        _read_return_file(LEGACY / f"{stem}_train.csv"),
        _read_return_file(LEGACY / f"{stem}_validation.csv"),
    ]
    if include_audit:
        pieces.append(_read_return_file(LEGACY / f"{stem}_audit.csv"))
    return pd.concat(pieces).sort_index()


def _v7_sleeve(source: Path, v2_output: Path, include_audit: bool) -> pd.Series:
    if not include_audit:
        return _read_return_file(V7, "session")
    panels, _ = load_panels(source, v2_output, AUDIT[1])
    artifact = fit_stress_artifact(panels["prior_market_volatility"], *TRAIN)
    result = run_candidate(
        panels["scores"]["core"], ResidualReversalConfig(), panels, artifact,
        PortfolioCosts(), AUDIT[1]
    )
    if result.status != "COMPLETED":
        raise RuntimeError(f"v7 R01 audit replay failed: {result.reason}")
    return result.daily.net_return


def _spy_returns(source: Path, sessions: pd.DatetimeIndex) -> pd.Series:
    bars = pd.read_csv(source / "bars_all.csv", usecols=["symbol", "t", "c"])
    bars = bars.loc[bars.symbol.eq("SPY")].copy()
    bars["session"] = (
        pd.to_datetime(bars.t, utc=True)
        .dt.tz_convert("America/New_York")
        .dt.tz_localize(None)
        .dt.normalize()
    )
    close = bars.set_index("session").c.astype(float).sort_index()
    return close.pct_change(fill_method=None).reindex(sessions).rename("SPY")


def load_sleeve_returns(
    source: Path, v2_output: Path, *, include_audit: bool
) -> tuple[pd.DataFrame, pd.Series, dict[str, str]]:
    mapping = {
        "legacy_reversal": _legacy_sleeve("reversal", include_audit),
        "kalman": _legacy_sleeve("kalman", include_audit),
        "dislocation": _legacy_sleeve("dislocation", include_audit),
        "legacy_blend": _legacy_sleeve("blend", include_audit),
        "v7_R01": _v7_sleeve(source, v2_output, include_audit),
    }
    end = AUDIT[1] if include_audit else VALIDATION[1]
    returns = pd.concat(mapping, axis=1).loc[TRAIN[0] : end]
    if returns.isna().any().any():
        missing = returns.isna().sum()
        raise RuntimeError(f"sleeve return alignment contains missing values: {missing.to_dict()}")
    benchmark = _spy_returns(source, returns.index)
    if benchmark.isna().any():
        raise RuntimeError("SPY benchmark is missing aligned sessions")
    hashes = {}
    for stem in ("reversal", "kalman", "dislocation", "blend"):
        for segment in ("train", "validation"):
            path = LEGACY / f"{stem}_{segment}.csv"
            hashes[str(path)] = sha256(path)
    hashes[str(V7)] = sha256(V7)
    return returns, benchmark, hashes


def segment_metrics(
    result: SleeveAllocationResult,
    benchmark: pd.Series,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> dict:
    if result.status != "COMPLETED":
        return {"status": result.status, "reason": result.reason}
    effective_start = max(start, result.evaluation_start)
    daily = result.daily.loc[effective_start:end]
    weights = result.weights.loc[daily.index]
    refits = result.refits.loc[effective_start:end]
    metrics = performance_metrics(daily.net_return)
    years = len(daily) / 252
    yearly = (1 + daily.net_return).groupby(daily.index.year).prod() - 1
    rolling_correlation = daily.net_return.rolling(63, min_periods=40).corr(
        benchmark.loc[daily.index]
    )
    metrics.update(
        status="COMPLETED",
        realized_market_correlation=float(daily.net_return.corr(benchmark.loc[daily.index])),
        maximum_abs_rolling_63d_correlation=float(rolling_correlation.abs().max()),
        annual_allocation_turnover=float(refits.turnover.sum() / years) if years else np.nan,
        total_allocation_cost=float(daily.allocation_cost.sum()),
        average_max_sleeve_weight=float(weights.max(axis=1).mean()),
        maximum_sleeve_weight=float(weights.max().max()),
        worst_calendar_year_return=float(yearly.min()),
        refits=int(len(refits)),
    )
    return metrics


def metric_deltas(candidate: dict, reference: dict) -> dict:
    keys = {
        "return": "cagr",
        "volatility": "annual_volatility",
        "correlation": "realized_market_correlation",
        "drawdown": "max_drawdown",
        "turnover": "annual_allocation_turnover",
    }
    return {f"delta_{name}_vs_S00": candidate[key] - reference[key] for name, key in keys.items()}


def main(source: Path, v2_output: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    sleeves, benchmark, source_hashes = load_sleeve_returns(
        source, v2_output, include_audit=False
    )
    base = SleeveAllocatorConfig()
    specifications = {
        "S00": replace(base, objective_mode="equal", return_floor="none"),
        "S01": replace(base, fit_window=126, hold_window=126, return_floor="spy"),
        "S02": replace(base, fit_window=63, hold_window=63, return_floor="spy"),
        "S03": replace(base, fit_window=126, hold_window=63, return_floor="spy"),
        "S04": replace(base, fit_window=126, hold_window=63, return_floor="zero"),
        "S05": replace(
            base, fit_window=126, hold_window=63, objective_mode="variance",
            return_floor="zero", variance_penalty=0.25, shrinkage_penalty=0.10,
        ),
        "S06": replace(
            base, fit_window=126, hold_window=63, objective_mode="downside",
            return_floor="zero", downside_penalty=0.25, shrinkage_penalty=0.10,
        ),
        "S07": replace(
            base, fit_window=126, hold_window=63, objective_mode="variance",
            return_floor="equal", variance_penalty=0.25, shrinkage_penalty=0.10,
        ),
    }
    freeze = {
        "protocol_sha256": sha256(PROTOCOL),
        "data_label": "research_snapshot_only",
        "source_hashes": source_hashes,
        "sleeves": list(SLEEVES),
        "candidate_configs": {name: asdict(config) for name, config in specifications.items()},
        "selection_data_end": str(VALIDATION[1].date()),
        "reused_audit_loaded_for_selection": False,
    }
    (output / "frozen_run_inputs.json").write_text(json.dumps(freeze, indent=2) + "\n")

    rows = []
    runs: dict[str, SleeveAllocationResult] = {}
    for name, config in specifications.items():
        result = run_sleeve_allocation(sleeves, benchmark, config, common_warmup=126)
        runs[name] = result
        train = segment_metrics(result, benchmark, *TRAIN)
        development = segment_metrics(result, benchmark, *VALIDATION)
        completed = result.status == "COMPLETED"
        eligible = completed and name != "S00" and (
            train["sharpe"] > 0.70
            and development["sharpe"] > 0.50
            and development["cagr"] > 0.05
            and train["max_drawdown"] >= -0.15
            and development["max_drawdown"] >= -0.15
            and development["annual_allocation_turnover"] <= 4
            and abs(train["realized_market_correlation"]) <= 0.20
            and abs(development["realized_market_correlation"]) <= 0.20
        )
        row = {
            "experiment": name,
            "status": "REFERENCE_ONLY" if name == "S00" and completed else result.status,
            "reason": result.reason,
            "eligible": eligible,
        }
        for segment_name, metrics in (("train", train), ("development", development)):
            row.update({f"{segment_name}_{key}": value for key, value in metrics.items()})
        if completed:
            row["robust_sharpe"] = min(train["sharpe"], development["sharpe"])
            row["worst_development_year"] = min(
                train["worst_calendar_year_return"], development["worst_calendar_year_return"]
            )
            result.daily.to_csv(output / f"{name}_daily.csv")
            result.refits.to_csv(output / f"{name}_refits.csv")
        rows.append(row)

    selection = pd.DataFrame(rows)
    reference = segment_metrics(runs["S00"], benchmark, *VALIDATION)
    delta_rows = []
    for name, result in runs.items():
        metrics = segment_metrics(result, benchmark, *VALIDATION)
        if metrics.get("status") == "COMPLETED":
            delta_rows.append({"experiment": name, **metric_deltas(metrics, reference)})
    pd.DataFrame(delta_rows).to_csv(output / "incremental_vs_S00.csv", index=False)
    selectable = selection.loc[selection.experiment.ne("S00")].sort_values(
        ["eligible", "robust_sharpe", "worst_development_year", "development_cagr",
         "development_annual_allocation_turnover", "experiment"],
        ascending=[False, False, False, False, True, True], na_position="last",
    )
    selection = pd.concat(
        [selection.loc[selection.experiment.eq("S00")], selectable], ignore_index=True
    )
    selection.to_csv(output / "candidate_selection.csv", index=False)
    if not selectable.eligible.any():
        selected_name = None
        selection_lock = {
            "status": "REJECTED",
            "reason": "NO_V8_CANDIDATE_PASSED_FROZEN_PROFITABILITY_GATES",
            "selected_before_reused_audit": None,
            "reused_audit_evaluated": False,
            "selection_sha256": sha256(output / "candidate_selection.csv"),
        }
    else:
        selected_name = str(selectable.loc[selectable.eligible].iloc[0].experiment)
        selection_lock = {
            "status": "SELECTED_PENDING_KILL_TESTS",
            "candidate": selected_name,
            "selected_before_reused_audit": True,
            "reused_audit_evaluated": False,
            "selection_sha256": sha256(output / "candidate_selection.csv"),
        }
    (output / "selection_lock.json").write_text(json.dumps(selection_lock, indent=2) + "\n")

    single_rows = []
    for sleeve in sleeves:
        for segment_name, (start, end) in (("train", TRAIN), ("development", VALIDATION)):
            series = sleeves[sleeve].loc[start:end]
            metrics = performance_metrics(series)
            single_rows.append(
                {
                    "sleeve": sleeve, "segment": segment_name, **metrics,
                    "market_correlation": float(series.corr(benchmark.loc[series.index])),
                }
            )
    pd.DataFrame(single_rows).to_csv(output / "single_sleeve_comparison.csv", index=False)

    kill_rows = []
    reused_audit = None
    if selected_name is None:
        pd.DataFrame(columns=["test", "passed", "status"]).to_csv(
            output / "kill_tests.csv", index=False
        )
        kill_lock = {"status": "NOT_RUN_NO_SELECTED_CANDIDATE"}
    else:
        selected_config = specifications[selected_name]
        tests = [
            ("base", sleeves, selected_config, 126, 0),
            ("double_allocation_cost", sleeves,
             replace(selected_config, allocation_cost_bps=10.0), 126, 0),
            ("implementation_delay_1", sleeves, selected_config, 126, 1),
            ("fit_window_minus_10", sleeves,
             replace(selected_config, fit_window=selected_config.fit_window - 10), 126, 0),
            ("fit_window_plus_10", sleeves,
             replace(selected_config, fit_window=selected_config.fit_window + 10), 136, 0),
        ]
        for sleeve in sleeves:
            tests.append(
                (f"remove_{sleeve}", sleeves.drop(columns=sleeve), selected_config, 126, 0)
            )
        for test_name, test_sleeves, config, warmup, delay in tests:
            result = run_sleeve_allocation(
                test_sleeves, benchmark, config, common_warmup=warmup,
                implementation_delay_sessions=delay,
            )
            metrics = segment_metrics(result, benchmark, *VALIDATION)
            passed = result.status == "COMPLETED" and metrics["sharpe"] > 0
            deltas = metric_deltas(metrics, reference) if result.status == "COMPLETED" else {}
            kill_rows.append(
                {"test": test_name, "passed": passed, "status": result.status,
                 **metrics, **deltas}
            )
        kill_table = pd.DataFrame(kill_rows)
        kill_table.to_csv(output / "kill_tests.csv", index=False)
        passed = bool(kill_table.passed.all())
        kill_lock = {
            "status": "PASSED" if passed else "REJECTED",
            "candidate": selected_name,
            "kill_tests_sha256": sha256(output / "kill_tests.csv"),
        }
    (output / "kill_test_lock.json").write_text(json.dumps(kill_lock, indent=2) + "\n")

    if selected_name is not None and kill_lock["status"] == "PASSED":
        audit_sleeves, audit_benchmark, _ = load_sleeve_returns(
            source, v2_output, include_audit=True
        )
        full = run_sleeve_allocation(
            audit_sleeves, audit_benchmark, specifications[selected_name], common_warmup=126
        )
        reused_audit = segment_metrics(full, audit_benchmark, *AUDIT)
        (output / "reused_audit.json").write_text(
            json.dumps(
                {"candidate": selected_name, "status": "REUSED_AUDIT_DESCRIPTIVE_ONLY",
                 "metrics": reused_audit},
                indent=2,
            )
            + "\n"
        )

    prospective = {
        "prospective_start": "2026-09-24",
        "status": "BLOCKED_PENDING_FRESH_DATA_AND_ACCEPTED_V8",
        "orders_allowed": False,
        "last_available_bar": "2026-09-18",
        "historical_rows_count_as_prospective": False,
    }
    (output / "prospective_shadow_start.json").write_text(
        json.dumps(prospective, indent=2) + "\n"
    )
    summary = {
        "protocol_sha256": sha256(PROTOCOL),
        "paper_objective": "MINIMIZE_ABSOLUTE_PORTFOLIO_MARKET_CORRELATION",
        "data_label": "research_snapshot_only",
        "selection": selection_lock,
        "kill_tests": kill_lock,
        "reused_audit": reused_audit,
        "prospective_shadow": prospective,
    }
    (output / "SUMMARY.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("output/equities_2026-09-20_sip_r2"))
    parser.add_argument("--v2-output", type=Path, default=Path("reports/equity_v2/backtest"))
    parser.add_argument("--output", type=Path, default=Path("reports/equity_v8"))
    args = parser.parse_args()
    main(args.source, args.v2_output, args.output)

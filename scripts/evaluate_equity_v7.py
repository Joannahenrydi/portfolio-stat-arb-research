"""Evaluate the frozen v7 residual short-horizon experiment matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd

from backtest.engine import performance_metrics
from backtest.residual_reversal import (
    ResidualReversalConfig,
    ResidualReversalResult,
    build_v7_scores,
    fit_stress_artifact,
    run_residual_reversal_backtest,
)
from features.alphas import AlphaConfig, build_equity_features, forward_total_return_labels
from portfolio.optimizer import PortfolioCosts
from scripts.evaluate_equity_v2 import AUDIT, TRAIN, VALIDATION, read_panel

PROTOCOL = Path("docs/EQUITY_RESEARCH_PROTOCOL_V7.md")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def maximum_zero_run(series: pd.Series) -> int:
    best = current = 0
    for value in series.fillna(0).le(0):
        current = current + 1 if value else 0
        best = max(best, current)
    return best


def daily_ic(signal: pd.DataFrame, label: pd.DataFrame, method: str) -> pd.Series:
    return signal.corrwith(label, axis=1, method=method)


def load_panels(source: Path, v2_output: Path, end: pd.Timestamp) -> tuple[dict, pd.DataFrame]:
    universe = pd.read_csv("reports/equity_v2/data_quality/prospective_400_stocks.csv")
    audit = pd.read_csv(v2_output / "backtest_universe_audit.csv")
    initial_symbols = audit.loc[audit.included, "symbol"].tolist()
    calendar = pd.read_csv(source / "market_calendar.csv")
    sessions = pd.DatetimeIndex(pd.to_datetime(calendar.date))
    sessions = sessions[(sessions >= "2017-01-03") & (sessions <= end)]
    raw_volume_initial = read_panel(source / "bars_raw.csv", initial_symbols, sessions, "v")
    zero_runs = raw_volume_initial.loc[TRAIN[0] : TRAIN[1]].apply(maximum_zero_run)
    excluded = set(zero_runs[zero_runs >= 20].index)
    symbols = [symbol for symbol in initial_symbols if symbol not in excluded]
    if len(symbols) != 341:
        raise RuntimeError(f"v7 expected the frozen 341-name cohort, received {len(symbols)}")
    sectors = universe.set_index("symbol").sector.reindex(symbols)
    adjusted = read_panel(source / "bars_all.csv", symbols + ["SPY"], sessions, "c")
    raw_close = read_panel(source / "bars_raw.csv", symbols, sessions, "c")
    raw_volume = raw_volume_initial[symbols]
    returns = adjusted[symbols].pct_change(fill_method=None)
    market = adjusted.SPY.pct_change(fill_method=None)
    sector_panel = pd.DataFrame(
        np.tile(sectors.to_numpy(), (len(returns), 1)), index=returns.index, columns=symbols
    )
    base_eligibility = returns.notna() & raw_close.notna() & raw_volume.notna() & raw_volume.gt(0)
    features = build_equity_features(
        returns, market, raw_volume, sector_panel, base_eligibility, AlphaConfig()
    )
    dollar_volume = raw_close * raw_volume
    adv = dollar_volume.rolling(60, min_periods=60).median().shift(1)
    liquidity_rank = adv.rank(axis=1, method="min", ascending=False)
    eligibility = (
        base_eligibility
        & adv.ge(20_000_000)
        & liquidity_rank.le(250)
        & sector_panel.notna()
    )
    residual_volatility = features.residual.rolling(60, min_periods=60).std().shift(1)
    prior_market_volatility = market.rolling(20, min_periods=20).std().shift(1) * np.sqrt(252)
    prior_dispersion = features.residual.std(axis=1).shift(1)
    prior_drawdown = adjusted.SPY.div(adjusted.SPY.cummax()).sub(1).shift(1)
    scores = build_v7_scores(features.residual, eligibility)
    panels = {
        "returns": returns,
        "market": market,
        "beta": features.market_beta_prior,
        "sectors": sectors,
        "raw_close": raw_close,
        "raw_volume": raw_volume,
        "adv": adv,
        "residual": features.residual,
        "residual_volatility": residual_volatility,
        "eligibility": eligibility,
        "prior_market_volatility": prior_market_volatility,
        "prior_dispersion": prior_dispersion,
        "prior_drawdown": prior_drawdown,
        "scores": scores,
        "symbols": symbols,
    }
    universe_audit = pd.DataFrame(
        {
            "symbol": zero_runs.index,
            "train_max_consecutive_zero_volume": zero_runs.to_numpy(),
            "included_v7": ~zero_runs.index.isin(excluded),
        }
    )
    return panels, universe_audit


def subset_panels(panels: dict, names: list[str]) -> dict:
    wide = {
        "returns", "beta", "raw_close", "raw_volume", "adv", "residual",
        "residual_volatility", "eligibility",
    }
    result = {key: value[names] if key in wide else value for key, value in panels.items()}
    result["sectors"] = panels["sectors"].reindex(names)
    result["scores"] = {key: value[names] for key, value in panels["scores"].items()}
    result["symbols"] = names
    return result


def run_candidate(
    score: pd.DataFrame,
    config: ResidualReversalConfig,
    panels: dict,
    stress_artifact,
    costs: PortfolioCosts,
    end: pd.Timestamp,
) -> ResidualReversalResult:
    index = panels["returns"].index
    mask = (index >= TRAIN[0]) & (index <= end)
    return run_residual_reversal_backtest(
        score.loc[mask], panels["returns"].loc[mask], panels["beta"].loc[mask],
        panels["sectors"], panels["raw_close"].loc[mask], panels["raw_volume"].loc[mask],
        panels["adv"].loc[mask], panels["residual_volatility"].loc[mask],
        panels["eligibility"].loc[mask], panels["prior_market_volatility"].loc[mask],
        stress_artifact, config=config, costs=costs,
    )


def segment_metrics(
    result: ResidualReversalResult,
    beta: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> dict:
    if result.status != "COMPLETED":
        return {"status": result.status, "reason": result.reason}
    daily = result.daily.loc[start:end]
    weights = result.weights.loc[daily.index]
    rebalances = result.rebalances.loc[start:end]
    net = performance_metrics(daily.net_return)
    gross = performance_metrics(daily.gross_return)
    yearly = (1 + daily.net_return).groupby(daily.index.year).prod() - 1
    years = len(daily) / 252
    net.update(
        status="COMPLETED",
        gross_sharpe=gross["sharpe"],
        gross_cagr=gross["cagr"],
        total_transaction_cost=float(daily.transaction_cost.sum()),
        total_borrow_cost=float(daily.borrow_cost.sum()),
        annual_turnover=float(daily.turnover.sum() / years),
        average_gross=float(daily.gross.mean()),
        average_abs_daily_net=float(daily.net.abs().mean()),
        average_abs_daily_beta=float((weights * beta.loc[daily.index]).sum(axis=1).abs().mean()),
        maximum_rebalance_net=float(rebalances.net_exposure.abs().max()),
        maximum_rebalance_beta=float(rebalances.beta_exposure.abs().max()),
        maximum_rebalance_sector=float(rebalances.max_sector_exposure.max()),
        worst_calendar_year_return=float(yearly.min()),
    )
    return net


def metric_deltas(candidate: dict, baseline: dict) -> dict:
    mapping = {
        "gross_alpha": "gross_cagr",
        "transaction_cost": "total_transaction_cost",
        "borrow": "total_borrow_cost",
        "turnover": "annual_turnover",
        "drawdown": "max_drawdown",
        "gross_exposure": "average_gross",
    }
    return {f"delta_{name}_vs_R01": candidate[key] - baseline[key] for name, key in mapping.items()}


def main(source: Path, v2_output: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    panels, universe_audit = load_panels(source, v2_output, VALIDATION[1])
    universe_audit.to_csv(output / "v7_universe_audit.csv", index=False)
    stress_artifact = fit_stress_artifact(panels["prior_market_volatility"], *TRAIN)
    base = ResidualReversalConfig()
    core = panels["scores"]["core"]
    specifications = {
        "R00": (panels["scores"]["reference"], base),
        "R01": (core, base),
        "R02": (core, replace(base, rebalance_every=2)),
        "R03": (core, replace(base, rebalance_every=5)),
        "R04": (core, replace(base, residual_vol_mode="mild")),
        "R05": (core, replace(base, residual_vol_mode="strong")),
        "R06": (core, replace(base, stress_scaler=True)),
        "R07": (core, replace(base, separate_side_ranks=True)),
        "R08": (core, replace(base, residual_vol_mode="mild", stress_scaler=True)),
    }
    freeze = {
        "protocol_sha256": sha256(PROTOCOL),
        "source_hashes": {
            "adjusted_bars": sha256(source / "bars_all.csv"),
            "raw_bars": sha256(source / "bars_raw.csv"),
        },
        "data_label": "research_snapshot_only",
        "cohort_size": len(panels["symbols"]),
        "development_end": str(VALIDATION[1].date()),
        "reused_audit_loaded_for_selection": False,
        "earnings_filter": "BLOCKED_MISSING_PIT_EARNINGS_CALENDAR",
        "borrow_ranking": "BLOCKED_MISSING_PIT_BORROW_DATA",
        "candidate_configs": {name: asdict(config) for name, (_, config) in specifications.items()},
        "stress_artifact": {
            "observations": len(stress_artifact.sorted_market_volatility),
            "sha256": hashlib.sha256(
                np.asarray(stress_artifact.sorted_market_volatility).tobytes()
            ).hexdigest(),
        },
    }
    (output / "frozen_run_inputs.json").write_text(json.dumps(freeze, indent=2) + "\n")

    costs = PortfolioCosts()
    runs: dict[str, ResidualReversalResult] = {}
    rows = []
    for name, (score, config) in specifications.items():
        result = run_candidate(score, config, panels, stress_artifact, costs, VALIDATION[1])
        runs[name] = result
        train = segment_metrics(result, panels["beta"], *TRAIN)
        development = segment_metrics(result, panels["beta"], *VALIDATION)
        completed = result.status == "COMPLETED"
        eligible = completed and name != "R00" and (
            train["sharpe"] > 0.70
            and development["sharpe"] > 0.50
            and development["cagr"] > 0.05
            and train["max_drawdown"] >= -0.15
            and development["max_drawdown"] >= -0.15
            and development["annual_turnover"] <= 25
            and max(
                development["maximum_rebalance_net"],
                development["maximum_rebalance_beta"],
                development["maximum_rebalance_sector"],
            ) <= base.neutrality_tolerance
        )
        row = {
            "experiment": name,
            "status": "REFERENCE_ONLY" if name == "R00" and completed else result.status,
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
        rows.append(row)
    selection = pd.DataFrame(rows)
    r01_metrics = segment_metrics(runs["R01"], panels["beta"], *VALIDATION)
    delta_rows = []
    for name, result in runs.items():
        metrics = segment_metrics(result, panels["beta"], *VALIDATION)
        if metrics.get("status") == "COMPLETED":
            delta_rows.append({"experiment": name, **metric_deltas(metrics, r01_metrics)})
    pd.DataFrame(delta_rows).to_csv(output / "incremental_vs_R01.csv", index=False)
    selectable = selection.loc[selection.experiment.ne("R00")].sort_values(
        ["eligible", "robust_sharpe", "worst_development_year", "development_cagr",
         "development_annual_turnover", "experiment"],
        ascending=[False, False, False, False, True, True], na_position="last",
    )
    selection = pd.concat(
        [selection.loc[selection.experiment.eq("R00")], selectable], ignore_index=True
    )
    selection.to_csv(output / "candidate_selection.csv", index=False)
    if not selectable.eligible.any():
        selected_name = None
        selection_lock = {
            "status": "REJECTED",
            "reason": "NO_V7_CANDIDATE_PASSED_FROZEN_PROFITABILITY_GATES",
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

    diagnostic_rows, side_rows = [], []
    for horizon in (1, 2, 3, 5, 10, 21):
        labels, label_end = forward_total_return_labels(panels["returns"], horizon)
        signal = core
        percentiles = signal.rank(axis=1, pct=True)
        for segment_name, (start, end) in (("train", TRAIN), ("development", VALIDATION)):
            completed = label_end.le(end)
            segment_signal = signal.where(completed).loc[start:end]
            segment_label = labels.where(completed).loc[start:end]
            segment_percentiles = percentiles.loc[start:end]
            linear = daily_ic(segment_signal, segment_label, "pearson")
            rank = daily_ic(segment_signal, segment_label, "spearman")
            long_mask = segment_percentiles.gt(0.5)
            short_mask = segment_percentiles.le(0.5)
            long_ic = daily_ic(
                segment_signal.where(long_mask), segment_label.where(long_mask), "spearman"
            )
            short_ic = daily_ic(
                segment_signal.where(short_mask), segment_label.where(short_mask), "spearman"
            )
            diagnostic_rows.append(
                {"horizon": horizon, "segment": segment_name,
                 "mean_ic": float(linear.mean()), "mean_rank_ic": float(rank.mean())}
            )
            side_rows.append(
                {"horizon": horizon, "segment": segment_name,
                 "long_rank_ic": float(long_ic.mean()), "short_rank_ic": float(short_ic.mean())}
            )
    pd.DataFrame(diagnostic_rows).to_csv(output / "signal_decay.csv", index=False)
    pd.DataFrame(side_rows).to_csv(output / "long_short_ic.csv", index=False)

    kill_rows = []
    reused_audit = None
    if selected_name is None:
        pd.DataFrame(columns=["test", "passed", "status"]).to_csv(
            output / "kill_tests.csv", index=False
        )
        kill_lock = {"status": "NOT_RUN_NO_SELECTED_CANDIDATE"}
    else:
        selected_score, selected_config = specifications[selected_name]
        contribution = (
            runs[selected_name].weights.loc[VALIDATION[0] : VALIDATION[1]]
            * panels["returns"].loc[VALIDATION[0] : VALIDATION[1]]
        ).sum()
        top_count = max(1, int(np.ceil(len(panels["symbols"]) * 0.05)))
        removals = {
            "remove_top_5pct_contributors": set(contribution.nlargest(top_count).index),
            "remove_deterministic_20pct": {
                symbol for symbol in panels["symbols"]
                if int(hashlib.sha256(symbol.encode()).hexdigest()[:8], 16) % 5 == 0
            },
        }
        tests = [
            ("base", selected_score, selected_config, panels, costs),
            (
                "double_cost", selected_score, selected_config, panels,
                replace(costs, multiplier=2.0),
            ),
            ("signal_delay_1", selected_score.shift(1), selected_config, panels, costs),
        ]
        for interval in (2, 3, 5):
            if interval != selected_config.rebalance_every:
                tests.append(
                    (f"rebalance_{interval}", selected_score,
                     replace(selected_config, rebalance_every=interval), panels, costs)
                )
        for test_name, removed in removals.items():
            kept = [symbol for symbol in panels["symbols"] if symbol not in removed]
            subset = subset_panels(panels, kept)
            tests.append((test_name, selected_score[kept], selected_config, subset, costs))
        for test_name, score, config, test_panels, test_costs in tests:
            result = run_candidate(
                score, config, test_panels, stress_artifact, test_costs, VALIDATION[1]
            )
            metrics = segment_metrics(result, test_panels["beta"], *VALIDATION)
            passed = result.status == "COMPLETED" and (
                metrics["sharpe"] > 0 if test_name == "double_cost" else metrics["total_return"] > 0
            )
            kill_rows.append(
                {"test": test_name, "passed": passed, "status": result.status,
                 **metrics, **metric_deltas(metrics, r01_metrics)}
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
        # This is deliberately after both locks are durable. The audit cannot change
        # selection, and it remains descriptive because earlier work already exposed it.
        audit_panels, _ = load_panels(source, v2_output, AUDIT[1])
        audit_artifact = fit_stress_artifact(audit_panels["prior_market_volatility"], *TRAIN)
        selected_config = specifications[selected_name][1]
        full = run_candidate(
            audit_panels["scores"]["core"], selected_config, audit_panels,
            audit_artifact, costs, AUDIT[1]
        )
        reused_audit = segment_metrics(full, audit_panels["beta"], *AUDIT)
        (output / "reused_audit.json").write_text(
            json.dumps(
                {
                    "candidate": selected_name,
                    "status": "REUSED_AUDIT_DESCRIPTIVE_ONLY",
                    "metrics": reused_audit,
                },
                indent=2,
            )
            + "\n"
        )

    prospective = {
        "prospective_start": "2026-09-24",
        "status": "BLOCKED_PENDING_FRESH_DATA_AND_ACCEPTED_V7",
        "orders_allowed": False,
        "last_available_bar": "2026-09-18",
        "historical_rows_count_as_prospective": False,
    }
    (output / "prospective_shadow_start.json").write_text(
        json.dumps(prospective, indent=2) + "\n"
    )
    summary = {
        "protocol_sha256": sha256(PROTOCOL),
        "data_label": "research_snapshot_only",
        "cohort_size": len(panels["symbols"]),
        "selection": selection_lock,
        "kill_tests": kill_lock,
        "reused_audit": reused_audit,
        "earnings_filter": "BLOCKED_MISSING_PIT_EARNINGS_CALENDAR",
        "borrow_aware_ranking": "BLOCKED_MISSING_PIT_BORROW_DATA",
        "prospective_shadow": prospective,
    }
    (output / "SUMMARY.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("output/equities_2026-09-20_sip_r2"))
    parser.add_argument("--v2-output", type=Path, default=Path("reports/equity_v2/backtest"))
    parser.add_argument("--output", type=Path, default=Path("reports/equity_v7"))
    args = parser.parse_args()
    main(args.source, args.v2_output, args.output)

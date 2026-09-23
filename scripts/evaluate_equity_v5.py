"""Run the frozen v5 conditional market-neutral experiment matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd

from backtest.conditional import (
    ConditionalBacktestResult,
    ConditionalConfig,
    fit_conditioning_artifact,
    run_conditional_backtest,
)
from backtest.engine import candidate_scores, performance_metrics, residual_returns
from features.alphas import (
    AlphaConfig,
    build_equity_features,
    fit_expected_return_calibration,
    forward_total_return_labels,
)
from portfolio.optimizer import PortfolioCosts
from scripts.evaluate_equity_v2 import AUDIT, TRAIN, VALIDATION, read_panel

PROTOCOL = Path("docs/EQUITY_RESEARCH_PROTOCOL_V5.md")


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


def cross_sectional_zscore(frame: pd.DataFrame) -> pd.DataFrame:
    ranks = frame.rank(axis=1, pct=True) - 0.5
    return ranks.div(ranks.std(axis=1, ddof=1).where(lambda value: value > 0), axis=0)


def daily_ic(signal: pd.DataFrame, label: pd.DataFrame, method: str = "spearman") -> pd.Series:
    return signal.corrwith(label, axis=1, method=method)


def apply_calibration(signals: dict[str, pd.DataFrame], artifact) -> pd.DataFrame:
    ordered = {name: signals[name] for name in artifact.families}
    first = next(iter(ordered.values()))
    values = np.stack([frame.to_numpy() for frame in ordered.values()], axis=-1)
    prediction = artifact.intercept + (
        (values - np.asarray(artifact.feature_means)) / np.asarray(artifact.feature_scales)
    ) @ np.asarray(artifact.coefficients)
    prediction[~np.isfinite(values).all(axis=2)] = np.nan
    return pd.DataFrame(prediction, index=first.index, columns=first.columns)


def segment_metrics(
    result: ConditionalBacktestResult,
    beta: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> dict:
    if result.status != "COMPLETED":
        return {"status": result.status, "reason": result.reason}
    daily = result.daily.loc[start:end]
    weights = result.weights.loc[daily.index]
    base = performance_metrics(daily.net_return)
    gross = performance_metrics(daily.gross_return)
    years = len(daily) / 252
    rebalances = result.rebalances.loc[start:end]
    yearly = (1 + daily.net_return).groupby(daily.index.year).prod() - 1
    base.update(
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
    return base


def run_candidate(
    score: pd.DataFrame,
    config: ConditionalConfig,
    panels: dict,
    artifact,
    costs: PortfolioCosts,
    end: pd.Timestamp,
) -> ConditionalBacktestResult:
    mask = (panels["returns"].index >= TRAIN[0]) & (panels["returns"].index <= end)
    return run_conditional_backtest(
        score.loc[mask],
        panels["returns"].loc[mask],
        panels["beta"].loc[mask],
        panels["sectors"],
        panels["raw_close"].loc[mask],
        panels["raw_volume"].loc[mask],
        panels["residual_volatility"].loc[mask],
        panels["prior_dispersion"].loc[mask],
        panels["prior_drawdown"].loc[mask],
        artifact,
        config=config,
        costs=costs,
    )


def subset_panels(panels: dict, names: list[str]) -> dict:
    wide = {"returns", "beta", "raw_close", "raw_volume", "residual_volatility"}
    result = {
        key: value[names] if key in wide else value
        for key, value in panels.items()
    }
    result["sectors"] = panels["sectors"].reindex(names)
    result["prior_dispersion"] = panels["residual"][names].std(axis=1).shift(1)
    result["residual"] = panels["residual"][names]
    return result


def main(source: Path, v2_output: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    initial = pd.read_csv("reports/equity_v2/data_quality/prospective_400_stocks.csv")
    audit = pd.read_csv(v2_output / "backtest_universe_audit.csv")
    initial_symbols = audit.loc[audit.included, "symbol"].tolist()
    calendar = pd.read_csv(source / "market_calendar.csv")
    sessions = pd.DatetimeIndex(pd.to_datetime(calendar.date))
    sessions = sessions[(sessions >= "2017-01-03") & (sessions <= AUDIT[1])]
    raw_volume_initial = read_panel(source / "bars_raw.csv", initial_symbols, sessions, "v")
    zero_runs = raw_volume_initial.loc[TRAIN[0] : TRAIN[1]].apply(maximum_zero_run)
    excluded = zero_runs[zero_runs >= 20].sort_index()
    symbols = [symbol for symbol in initial_symbols if symbol not in set(excluded.index)]
    if symbols != [symbol for symbol in initial_symbols if symbol not in {"COR", "DOW", "SNDK"}]:
        raise RuntimeError("train-only stale gate differs from the frozen v5 cohort")
    universe = initial[initial.symbol.isin(symbols)].copy()
    sectors = universe.set_index("symbol").sector.reindex(symbols)
    adjusted = read_panel(source / "bars_all.csv", symbols + ["SPY"], sessions, "c")
    raw_close = read_panel(source / "bars_raw.csv", symbols, sessions, "c")
    raw_volume = raw_volume_initial[symbols]
    returns = adjusted[symbols].pct_change(fill_method=None)
    market = adjusted.SPY.pct_change(fill_method=None)
    residual, beta = residual_returns(returns, market, sectors)
    baseline = candidate_scores(residual)["momentum_252_skip_21"]
    residual_volatility = residual.rolling(60, min_periods=60).std().shift(1)
    prior_dispersion = residual.std(axis=1).shift(1)
    prior_drawdown = adjusted.SPY.div(adjusted.SPY.cummax()).sub(1).shift(1)
    conditioning = fit_conditioning_artifact(prior_dispersion, *TRAIN)
    panels = {
        "returns": returns,
        "beta": beta,
        "sectors": sectors,
        "raw_close": raw_close,
        "raw_volume": raw_volume,
        "residual": residual,
        "residual_volatility": residual_volatility,
        "prior_dispersion": prior_dispersion,
        "prior_drawdown": prior_drawdown,
    }
    pd.DataFrame(
        {
            "symbol": zero_runs.index,
            "train_max_consecutive_zero_volume": zero_runs.to_numpy(),
        }
    ).assign(included_v5=lambda frame: frame.train_max_consecutive_zero_volume.lt(20)).to_csv(
        output / "v5_universe_audit.csv", index=False
    )

    sector_panel = pd.DataFrame(
        np.tile(sectors.to_numpy(), (len(returns), 1)), index=returns.index, columns=symbols
    )
    eligibility = returns.notna() & raw_volume.notna() & raw_volume.gt(0)
    features = build_equity_features(
        returns,
        market,
        raw_volume,
        sector_panel,
        eligibility,
        AlphaConfig(),
    )
    families = {"baseline_momentum": baseline, **features.signals}
    forward5, label_end5 = forward_total_return_labels(returns, 5)
    train_mask = (returns.index >= TRAIN[0]) & (returns.index <= TRAIN[1])
    ic_series = {
        name: daily_ic(signal, forward5).loc[train_mask]
        for name, signal in families.items()
    }
    family_rows = []
    admitted = ["baseline_momentum"]
    for name in ("residual_reversal", "kalman_innovation", "volume_volatility_dislocation"):
        mean_ic = float(ic_series[name].mean())
        max_correlation = max(
            abs(float(ic_series[name].corr(ic_series[other]))) for other in admitted
        )
        qualified = mean_ic > 0 and max_correlation < 0.75
        family_rows.append(
            {
                "family": name,
                "train_mean_rank_ic_5d": mean_ic,
                "max_admitted_ic_correlation": max_correlation,
                "qualified": qualified,
            }
        )
        if qualified:
            admitted.append(name)
    family_rows.insert(
        0,
        {
            "family": "baseline_momentum",
            "train_mean_rank_ic_5d": float(ic_series["baseline_momentum"].mean()),
            "max_admitted_ic_correlation": 0.0,
            "qualified": True,
        },
    )
    pd.DataFrame(family_rows).to_csv(output / "alpha_family_qualification.csv", index=False)
    ic_matrix = pd.DataFrame(ic_series).corr()
    ic_matrix.to_csv(output / "alpha_ic_correlation_train.csv")

    ensemble_scores: dict[str, pd.DataFrame] = {}
    ensemble_status = "AVAILABLE" if len(admitted) >= 2 else "BLOCKED_FEWER_THAN_TWO_FAMILIES"
    if len(admitted) >= 2:
        inverse_volatility = {
            name: 1 / ic_series[name].std(ddof=1) for name in admitted
        }
        total = sum(inverse_volatility.values())
        weights = {name: value / total for name, value in inverse_volatility.items()}
        ensemble_scores["E13"] = sum(
            weights[name] * cross_sectional_zscore(families[name]) for name in admitted
        )
        artifact = fit_expected_return_calibration(
            {name: families[name] for name in admitted},
            forward5,
            label_end5,
            train_start=TRAIN[0],
            train_end=TRAIN[1],
            pit_verified=False,
            input_scope="research_snapshot_only",
            min_samples=100_000,
        )
        ensemble_scores["E14"] = apply_calibration(
            {name: families[name] for name in admitted}, artifact
        )
        (output / "ensemble_artifacts.json").write_text(
            json.dumps(
                {
                    "status": ensemble_status,
                    "admitted_families": admitted,
                    "equal_risk_weights": weights,
                    "ridge_families": list(artifact.families),
                    "ridge_coefficients": list(artifact.coefficients),
                    "ridge_intercept": artifact.intercept,
                    "training_digest": artifact.training_digest,
                    "latest_training_label": str(artifact.latest_label_session.date()),
                },
                indent=2,
            )
            + "\n"
        )
    else:
        (output / "ensemble_artifacts.json").write_text(
            json.dumps({"status": ensemble_status, "admitted_families": admitted}, indent=2) + "\n"
        )

    base = ConditionalConfig()
    specifications = {
        "E01": (baseline, base),
        "E02": (baseline, replace(base, residual_vol_mode="mild")),
        "E03": (baseline, replace(base, residual_vol_mode="strong")),
        "E04": (baseline, replace(base, residual_vol_mode="capped")),
        "E05": (baseline, replace(base, short_multiplier=0.75)),
        "E06": (baseline, replace(base, short_multiplier=0.50)),
        "E07": (baseline, replace(base, regime_mode="mild")),
        "E08": (baseline, replace(base, regime_mode="defensive")),
        "E09": (baseline, replace(base, residual_vol_mode="mild", regime_mode="mild")),
        "E10": (baseline, replace(base, residual_vol_mode="mild", short_multiplier=0.75)),
        "E11": (baseline, replace(base, short_multiplier=0.75, regime_mode="mild")),
        "E12": (
            baseline,
            replace(base, residual_vol_mode="mild", short_multiplier=0.75, regime_mode="mild"),
        ),
    }
    for name, score in ensemble_scores.items():
        specifications[name] = (score, base)
    freeze = {
        "protocol_sha256": sha256(PROTOCOL),
        "source_hashes": {
            "adjusted_bars": sha256(source / "bars_all.csv"),
            "raw_bars": sha256(source / "bars_raw.csv"),
        },
        "cohort_size": len(symbols),
        "stale_exclusions": excluded.to_dict(),
        "conditioning_artifact": {
            "train_start": str(conditioning.train_start.date()),
            "train_end": str(conditioning.train_end.date()),
            "observations": len(conditioning.sorted_dispersion),
            "sha256": hashlib.sha256(np.asarray(conditioning.sorted_dispersion).tobytes()).hexdigest(),
        },
        "candidate_configs": {name: asdict(config) for name, (_, config) in specifications.items()},
        "ensemble_status": ensemble_status,
        "audit_status": "REUSED_AUDIT_NOT_FOR_SELECTION",
    }
    (output / "frozen_run_inputs.json").write_text(json.dumps(freeze, indent=2) + "\n")

    baseline_evaluation = json.loads((v2_output / "evaluation.json").read_text())
    rows = [
        {
            "experiment": "E00",
            "status": "REFERENCE_ONLY",
            "eligible": False,
            "train_sharpe": baseline_evaluation["train"]["sharpe"],
            "validation_sharpe": baseline_evaluation["validation"]["sharpe"],
            "validation_cagr": baseline_evaluation["validation"]["cagr"],
            "reason": "UNMODIFIED_V2_BASELINE",
        }
    ]
    development_runs: dict[str, ConditionalBacktestResult] = {}
    costs = PortfolioCosts()
    for name in sorted(specifications):
        score, config = specifications[name]
        result = run_candidate(score, config, panels, conditioning, costs, VALIDATION[1])
        development_runs[name] = result
        train = segment_metrics(result, beta, *TRAIN)
        validation = segment_metrics(result, beta, *VALIDATION)
        completed = result.status == "COMPLETED"
        eligible_candidate = completed and (
            train["sharpe"] > 0
            and validation["sharpe"] > 0
            and validation["cagr"] > 0
            and validation["max_drawdown"] >= -0.15
            and validation["annual_turnover"] <= 25
            and max(
                validation["maximum_rebalance_net"],
                validation["maximum_rebalance_beta"],
                validation["maximum_rebalance_sector"],
            ) <= base.neutrality_tolerance
        )
        row = {
            "experiment": name,
            "status": result.status,
            "reason": result.reason,
            "eligible": eligible_candidate,
        }
        for segment_name, metrics in (("train", train), ("validation", validation)):
            row.update({f"{segment_name}_{key}": value for key, value in metrics.items()})
        if completed:
            row["robust_sharpe"] = min(train["sharpe"], validation["sharpe"])
            row["worst_development_year"] = min(
                train["worst_calendar_year_return"], validation["worst_calendar_year_return"]
            )
        rows.append(row)
    selection = pd.DataFrame(rows)
    selectable = selection.loc[selection.experiment.ne("E00")].copy()
    selectable = selectable.sort_values(
        ["eligible", "robust_sharpe", "worst_development_year", "validation_sharpe",
         "validation_annual_turnover", "experiment"],
        ascending=[False, False, False, False, True, True],
        na_position="last",
    )
    selection = pd.concat([selection.loc[selection.experiment.eq("E00")], selectable], ignore_index=True)
    selection.to_csv(output / "candidate_selection.csv", index=False)
    if not selectable.eligible.any():
        lock = {
            "status": "REJECTED",
            "reason": "NO_V5_CANDIDATE_PASSED_FROZEN_DEVELOPMENT_GATES",
            "selected_before_reused_audit": None,
            "reused_audit_evaluated": False,
            "selection_sha256": sha256(output / "candidate_selection.csv"),
        }
        (output / "selection_lock.json").write_text(json.dumps(lock, indent=2) + "\n")
        selected_name = None
    else:
        selected_name = str(selectable.loc[selectable.eligible].iloc[0].experiment)
        lock = {
            "status": "SELECTED_PENDING_KILL_TESTS",
            "candidate": selected_name,
            "selected_before_reused_audit": True,
            "reused_audit_evaluated": False,
            "selection_sha256": sha256(output / "candidate_selection.csv"),
        }
        (output / "selection_lock.json").write_text(json.dumps(lock, indent=2) + "\n")

    decay_rows = []
    long_short_rows = []
    for horizon in (1, 5, 10, 21, 42):
        labels, _ = forward_total_return_labels(returns, horizon)
        linear = daily_ic(baseline, labels, "pearson")
        rank_ic = daily_ic(baseline, labels, "spearman")
        percentiles = baseline.rank(axis=1, pct=True)
        long_ic = daily_ic(baseline.where(percentiles.gt(0.5)), labels.where(percentiles.gt(0.5)))
        short_ic = daily_ic(baseline.where(percentiles.le(0.5)), labels.where(percentiles.le(0.5)))
        for segment_name, (start, end) in (("train", TRAIN), ("validation", VALIDATION)):
            decay_rows.append(
                {
                    "horizon": horizon,
                    "segment": segment_name,
                    "mean_ic": float(linear.loc[start:end].mean()),
                    "mean_rank_ic": float(rank_ic.loc[start:end].mean()),
                }
            )
            long_short_rows.append(
                {
                    "horizon": horizon,
                    "segment": segment_name,
                    "long_rank_ic": float(long_ic.loc[start:end].mean()),
                    "short_rank_ic": float(short_ic.loc[start:end].mean()),
                }
            )
    pd.DataFrame(decay_rows).to_csv(output / "signal_decay.csv", index=False)
    pd.DataFrame(long_short_rows).to_csv(output / "long_short_ic.csv", index=False)

    kill_rows = []
    reused_audit = None
    if selected_name is not None:
        selected_score, selected_config = specifications[selected_name]
        selected_result = development_runs[selected_name]
        validation_contribution = (
            selected_result.weights.loc[VALIDATION[0] : VALIDATION[1]]
            * returns.loc[VALIDATION[0] : VALIDATION[1]]
        ).sum()
        top_count = max(1, int(np.ceil(len(symbols) * 0.05)))
        remove_top = set(validation_contribution.nlargest(top_count).index)
        deterministic_remove = {
            symbol for symbol in symbols
            if int(hashlib.sha256(symbol.encode()).hexdigest()[:8], 16) % 5 == 0
        }
        tests = [
            ("base", selected_score, selected_config, panels, costs, conditioning),
            ("double_cost", selected_score, selected_config, panels,
             replace(costs, multiplier=2.0), conditioning),
            ("signal_delay_1", selected_score.shift(1), selected_config, panels, costs, conditioning),
            ("conditioning_slope_90pct", selected_score,
             replace(selected_config, conditioning_strength=0.9), panels, costs, conditioning),
            ("conditioning_slope_110pct", selected_score,
             replace(selected_config, conditioning_strength=1.1), panels, costs, conditioning),
            ("rebalance_10", selected_score,
             replace(selected_config, rebalance_every=10), panels, costs, conditioning),
        ]
        for test_name, removed in (
            ("remove_deterministic_20pct", deterministic_remove),
            ("remove_top_5pct_contributors", remove_top),
        ):
            kept = [symbol for symbol in symbols if symbol not in removed]
            test_panels = subset_panels(panels, kept)
            test_artifact = fit_conditioning_artifact(test_panels["prior_dispersion"], *TRAIN)
            tests.append(
                (test_name, selected_score[kept], selected_config, test_panels, costs, test_artifact)
            )
        for test_name, test_score, test_config, test_panels, test_costs, test_artifact in tests:
            result = run_candidate(
                test_score, test_config, test_panels, test_artifact, test_costs, VALIDATION[1]
            )
            metrics = segment_metrics(result, test_panels["beta"], *VALIDATION)
            passed = result.status == "COMPLETED" and metrics["total_return"] > 0
            kill_rows.append(
                {"test": test_name, "passed": passed, "status": result.status, **metrics}
            )
        kill_table = pd.DataFrame(kill_rows)
        kill_table.to_csv(output / "kill_tests.csv", index=False)
        kill_passed = bool(kill_table.passed.all())
        kill_lock = {
            "status": "PASSED" if kill_passed else "REJECTED",
            "candidate": selected_name,
            "all_validation_total_returns_positive": kill_passed,
            "kill_tests_sha256": sha256(output / "kill_tests.csv"),
        }
        (output / "kill_test_lock.json").write_text(json.dumps(kill_lock, indent=2) + "\n")
        if kill_passed:
            full = run_candidate(
                selected_score, selected_config, panels, conditioning, costs, AUDIT[1]
            )
            reused_audit = segment_metrics(full, beta, *AUDIT)
            (output / "reused_audit.json").write_text(
                json.dumps(
                    {
                        "candidate": selected_name,
                        "status": "REUSED_AUDIT_NOT_FOR_SELECTION",
                        "metrics": reused_audit,
                    },
                    indent=2,
                )
                + "\n"
            )
            full.daily.to_csv(output / "selected_daily.csv")
            full.weights.loc[AUDIT[0] :].to_csv(
                output / "selected_weights_reused_audit.csv.gz", compression="gzip"
            )
    else:
        pd.DataFrame(columns=["test", "passed", "status"]).to_csv(
            output / "kill_tests.csv", index=False
        )
        (output / "kill_test_lock.json").write_text(
            json.dumps({"status": "NOT_RUN_NO_SELECTED_CANDIDATE"}, indent=2) + "\n"
        )

    prospective = {
        "prospective_start": "2026-09-24",
        "status": "BLOCKED_PENDING_FRESH_DATA_AND_ACCEPTED_V5",
        "orders_allowed": False,
        "last_available_bar": "2026-09-18",
        "credentials_configured_in_run": False,
        "historical_rows_count_as_prospective": False,
    }
    (output / "prospective_shadow_start.json").write_text(
        json.dumps(prospective, indent=2) + "\n"
    )
    summary = {
        "protocol_sha256": sha256(PROTOCOL),
        "cohort_size": len(symbols),
        "selection": lock,
        "reused_audit": reused_audit,
        "ensemble_status": ensemble_status,
        "borrow_aware_ranking": "BLOCKED_MISSING_PIT_BORROW_DATA",
        "prospective_shadow": prospective,
    }
    (output / "SUMMARY.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("output/equities_2026-09-20_sip_r2"))
    parser.add_argument("--v2-output", type=Path, default=Path("reports/equity_v2/backtest"))
    parser.add_argument("--output", type=Path, default=Path("reports/equity_v5"))
    args = parser.parse_args()
    main(args.source, args.v2_output, args.output)

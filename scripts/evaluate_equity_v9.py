"""Evaluate the pre-registered v9 adaptive, orthogonal, cost-gated alpha matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd

from backtest.residual_reversal import ResidualReversalConfig, fit_stress_artifact
from features.adaptive_alpha import (
    adaptive_kalman_innovation,
    apply_confidence_filter,
    blend_families,
    cost_hurdle,
    expected_return,
    fit_icir_blend,
    fit_zero_intercept_calibration,
    orthogonalize_families,
    remove_sector_peer_mean,
)
from features.alphas import forward_total_return_labels
from portfolio.optimizer import PortfolioCosts
from scripts.evaluate_equity_v2 import TRAIN, VALIDATION
from scripts.evaluate_equity_v7 import (
    load_panels,
    run_candidate,
    segment_metrics as base_segment_metrics,
    sha256,
    subset_panels,
)

PROTOCOL = Path("docs/EQUITY_RESEARCH_PROTOCOL_V9.md")
FAMILY_ORDER = (
    "adaptive_kalman",
    "residual_reversal",
    "volume_price_dislocation",
    "volatility_conditioned_reversal",
)


def json_ready(value):
    if isinstance(value, pd.Timestamp):
        return str(value.date())
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    if isinstance(value, tuple):
        return [json_ready(item) for item in value]
    if isinstance(value, dict):
        return {key: json_ready(item) for key, item in value.items()}
    return value


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(json_ready(value), indent=2, allow_nan=False) + "\n")


def standardize(frame: pd.DataFrame, eligibility: pd.DataFrame) -> pd.DataFrame:
    usable = frame.where(eligibility)
    centered = usable.sub(usable.mean(axis=1), axis=0)
    scale = usable.std(axis=1, ddof=1).where(usable.count(axis=1).ge(2))
    return centered.div(scale.where(scale.gt(0)), axis=0).clip(-3, 3)


def build_v9_artifacts(panels: dict) -> dict:
    eligibility = panels["eligibility"]
    r_adaptive = adaptive_kalman_innovation(
        panels["returns"], panels["market"], mode="r_adaptive"
    )
    qr_adaptive = adaptive_kalman_innovation(
        panels["returns"], panels["market"], mode="qr_adaptive"
    )
    r_signal = standardize(
        remove_sector_peer_mean(r_adaptive.signal, panels["sectors"], eligibility),
        eligibility,
    )
    qr_raw = remove_sector_peer_mean(qr_adaptive.signal, panels["sectors"], eligibility)
    residual3 = panels["residual"].rolling(3, min_periods=3).sum()
    vol20 = panels["residual"].rolling(20, min_periods=20).std().shift(1)
    vol60 = panels["residual_volatility"]
    vol_ratio = vol20.div(vol60.where(vol60.gt(0))).clip(0.5, 2.0)
    raw_families = {
        "adaptive_kalman": qr_raw,
        "residual_reversal": panels["feature_signals"]["residual_reversal"],
        "volume_price_dislocation": panels["feature_signals"][
            "volume_volatility_dislocation"
        ],
        "volatility_conditioned_reversal": -residual3 * vol_ratio,
    }
    orthogonal = orthogonalize_families(raw_families, eligibility)
    labels, label_end = forward_total_return_labels(panels["returns"], 3)
    icir = fit_icir_blend(orthogonal, labels, label_end, *TRAIN)
    icir_weights = dict(zip(icir.families, icir.weights))
    four_equal = blend_families(orthogonal, {name: 0.25 for name in FAMILY_ORDER})
    four_icir = blend_families(orthogonal, icir_weights)
    two_orthogonal = {name: orthogonal[name] for name in FAMILY_ORDER[:2]}
    two_equal = blend_families(
        two_orthogonal, {"adaptive_kalman": 0.5, "residual_reversal": 0.5}
    )
    calibration_four = fit_zero_intercept_calibration(
        four_icir, labels, label_end, *TRAIN
    )
    calibration_two = fit_zero_intercept_calibration(
        two_equal, labels, label_end, *TRAIN
    )
    daily_vol = panels["returns"].rolling(60, min_periods=60).std().shift(1)
    forecast_four = expected_return(four_icir, calibration_four)
    forecast_two = expected_return(two_equal, calibration_two)
    hurdle_four = cost_hurdle(forecast_four, daily_vol, panels["adv"])
    hurdle_two = cost_hurdle(forecast_two, daily_vol, panels["adv"])
    return {
        "r_adaptive": r_adaptive,
        "qr_adaptive": qr_adaptive,
        "r_signal": r_signal,
        "qr_signal": orthogonal["adaptive_kalman"],
        "orthogonal": orthogonal,
        "labels": labels,
        "label_end": label_end,
        "icir_artifact": icir,
        "four_equal": four_equal,
        "four_icir": four_icir,
        "two_equal": two_equal,
        "calibration_four": calibration_four,
        "calibration_two": calibration_two,
        "forecast_four": forecast_four,
        "forecast_two": forecast_two,
        "hurdle_four": hurdle_four,
        "hurdle_two": hurdle_two,
    }


def segment_metrics(result, beta, start, end) -> dict:
    metrics = base_segment_metrics(result, beta, start, end)
    if metrics.get("status") == "COMPLETED":
        rebalances = result.rebalances.loc[start:end]
        metrics["average_eligible_names"] = float(rebalances.eligible_names.mean())
        metrics["neutrality_failures"] = 0
    return metrics


def gate(train: dict, development: dict, tolerance: float) -> bool:
    return bool(
        train.get("status") == "COMPLETED"
        and development.get("status") == "COMPLETED"
        and train["sharpe"] > 0.70
        and development["sharpe"] > 0.50
        and development["cagr"] > 0.05
        and train["max_drawdown"] >= -0.15
        and development["max_drawdown"] >= -0.15
        and development["annual_turnover"] <= 25
        and development["average_eligible_names"] >= 100
        and max(
            development["maximum_rebalance_net"],
            development["maximum_rebalance_beta"],
            development["maximum_rebalance_sector"],
        ) <= tolerance
    )


def family_diagnostics(artifacts: dict, output: Path) -> None:
    rows = []
    for name, signal in artifacts["orthogonal"].items():
        for segment, (start, end) in (("train", TRAIN), ("development", VALIDATION)):
            complete = artifacts["label_end"].le(end)
            label = artifacts["labels"].where(complete).loc[start:end]
            usable_signal = signal.where(complete).loc[start:end]
            for year, subset in usable_signal.groupby(usable_signal.index.year):
                matching = label.loc[subset.index]
                rows.append(
                    {
                        "family": name,
                        "segment": segment,
                        "year": int(year),
                        "mean_ic": float(subset.corrwith(matching, axis=1).mean()),
                        "mean_rank_ic": float(
                            subset.corrwith(matching, axis=1, method="spearman").mean()
                        ),
                    }
                )
    pd.DataFrame(rows).to_csv(output / "family_ic_by_year.csv", index=False)
    family_returns = pd.DataFrame(
        {
            name: (signal * artifacts["labels"]).mean(axis=1)
            for name, signal in artifacts["orthogonal"].items()
        }
    )
    family_returns.loc[TRAIN[0] : TRAIN[1]].corr().to_csv(
        output / "train_family_return_correlation.csv"
    )


def qr_diagnostics(artifacts: dict, output: Path) -> None:
    rows = []
    for model in ("r_adaptive", "qr_adaptive"):
        result = artifacts[model]
        for field in ("q_prior", "r_prior"):
            frame = getattr(result, field)
            for segment, (start, end) in (("train", TRAIN), ("development", VALIDATION)):
                values = frame.loc[start:end].to_numpy().ravel()
                values = values[np.isfinite(values)]
                quantiles = np.quantile(values, [0, 0.01, 0.25, 0.5, 0.75, 0.99, 1])
                rows.append(
                    {
                        "model": model,
                        "field": field,
                        "segment": segment,
                        **{
                            key: float(value)
                            for key, value in zip(
                                ("min", "p01", "p25", "p50", "p75", "p99", "max"),
                                quantiles,
                            )
                        },
                    }
                )
    pd.DataFrame(rows).to_csv(output / "adaptive_qr_distributions.csv", index=False)


def filter_diagnostics(
    forecast: pd.DataFrame,
    hurdle: pd.DataFrame,
    passed: pd.DataFrame,
    multiplier: float,
    experiment: str,
) -> list[dict]:
    rows = []
    ratio = forecast.abs().div(hurdle.where(hurdle.gt(0)))
    for year in range(2018, 2025):
        mask = passed.index.year == year
        eligible_inputs = forecast.loc[mask].notna() & hurdle.loc[mask].notna()
        denominator = int(eligible_inputs.to_numpy().sum())
        rows.append(
            {
                "experiment": experiment,
                "multiplier": multiplier,
                "year": year,
                "observations": denominator,
                "pass_rate": float(passed.loc[mask].to_numpy().sum() / denominator)
                if denominator
                else np.nan,
                "median_edge_to_cost": float(ratio.loc[mask].where(passed.loc[mask]).stack().median()),
            }
        )
    return rows


def main(source: Path, v2_output: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    panels, universe_audit = load_panels(source, v2_output, VALIDATION[1])
    artifacts = build_v9_artifacts(panels)
    family_diagnostics(artifacts, output)
    qr_diagnostics(artifacts, output)
    write_json(output / "icir_artifact.json", asdict(artifacts["icir_artifact"]))
    write_json(
        output / "calibration_artifacts.json",
        {
            "four_family": asdict(artifacts["calibration_four"]),
            "two_family": asdict(artifacts["calibration_two"]),
        },
    )
    universe_audit.to_csv(output / "v9_universe_audit.csv", index=False)
    base = ResidualReversalConfig()
    stress_artifact = fit_stress_artifact(panels["prior_market_volatility"], *TRAIN)
    filtered: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    filter_rows = []
    for name, multiplier in (("K05", 1.0), ("K06", 1.5), ("K07", 2.0)):
        score, passed = apply_confidence_filter(
            artifacts["four_icir"], artifacts["forecast_four"],
            artifacts["hurdle_four"], multiplier,
        )
        filtered[name] = (score, passed)
        filter_rows.extend(
            filter_diagnostics(
                artifacts["forecast_four"], artifacts["hurdle_four"], passed,
                multiplier, name,
            )
        )
    score_k08, passed_k08 = apply_confidence_filter(
        artifacts["two_equal"], artifacts["forecast_two"], artifacts["hurdle_two"], 1.0
    )
    filtered["K08"] = (score_k08, passed_k08)
    filter_rows.extend(
        filter_diagnostics(
            artifacts["forecast_two"], artifacts["hurdle_two"], passed_k08, 1.0, "K08"
        )
    )
    pd.DataFrame(filter_rows).to_csv(output / "confidence_filter_by_year.csv", index=False)

    specifications = {
        "K00": panels["feature_signals"]["kalman_innovation"],
        "K01": artifacts["r_signal"],
        "K02": artifacts["qr_signal"],
        "K03": artifacts["four_equal"],
        "K04": artifacts["four_icir"],
        "K05": filtered["K05"][0],
        "K06": filtered["K06"][0],
        "K07": filtered["K07"][0],
        "K08": filtered["K08"][0],
    }
    freeze = {
        "protocol_sha256": sha256(PROTOCOL),
        "source_hashes": {
            "adjusted_bars": sha256(source / "bars_all.csv"),
            "raw_bars": sha256(source / "bars_raw.csv"),
        },
        "candidate_ids": list(specifications),
        "data_label": "research_snapshot_only",
        "cohort_size": len(panels["symbols"]),
        "development_end": str(VALIDATION[1].date()),
        "reused_audit_loaded_for_selection": False,
        "config": asdict(base),
        "costs": asdict(PortfolioCosts()),
        "earnings_filter": "BLOCKED_MISSING_PIT_EARNINGS_CALENDAR",
        "name_level_borrow": "BLOCKED_MISSING_PIT_BORROW_DATA",
    }
    write_json(output / "frozen_run_inputs.json", freeze)

    runs, rows = {}, []
    for name, score in specifications.items():
        result = run_candidate(
            score, base, panels, stress_artifact, PortfolioCosts(), VALIDATION[1]
        )
        runs[name] = result
        train = segment_metrics(result, panels["beta"], *TRAIN)
        development = segment_metrics(result, panels["beta"], *VALIDATION)
        eligible = name != "K00" and gate(train, development, base.neutrality_tolerance)
        row = {
            "experiment": name,
            "status": "REFERENCE_ONLY" if name == "K00" and result.status == "COMPLETED" else result.status,
            "reason": result.reason,
            "eligible": eligible,
        }
        row.update({f"train_{key}": value for key, value in train.items()})
        row.update({f"development_{key}": value for key, value in development.items()})
        if result.status == "COMPLETED":
            row["robust_sharpe"] = min(train["sharpe"], development["sharpe"])
            row["worst_calendar_year"] = min(
                train["worst_calendar_year_return"], development["worst_calendar_year_return"]
            )
        rows.append(row)
    selection = pd.DataFrame(rows)
    selectable = selection.loc[selection.experiment.ne("K00")].sort_values(
        ["eligible", "robust_sharpe", "worst_calendar_year", "development_cagr",
         "development_annual_turnover", "experiment"],
        ascending=[False, False, False, False, True, True], na_position="last",
    )
    selection = pd.concat(
        [selection.loc[selection.experiment.eq("K00")], selectable], ignore_index=True
    )
    selection.to_csv(output / "candidate_selection.csv", index=False)
    selected = (
        str(selectable.loc[selectable.eligible].iloc[0].experiment)
        if selectable.eligible.any()
        else None
    )
    selection_lock = {
        "status": "SELECTED_PENDING_KILL_TESTS" if selected else "REJECTED",
        "reason": None if selected else "NO_V9_CANDIDATE_PASSED_FROZEN_PROFITABILITY_GATES",
        "candidate": selected,
        "reused_audit_evaluated": False,
        "selection_sha256": sha256(output / "candidate_selection.csv"),
    }
    write_json(output / "selection_lock.json", selection_lock)

    kill_rows = []
    if selected is None:
        pd.DataFrame(columns=["test", "passed", "status"]).to_csv(
            output / "kill_tests.csv", index=False
        )
        kill_lock = {"status": "NOT_RUN_NO_SELECTED_CANDIDATE"}
    else:
        selected_score = specifications[selected]
        development_weights = runs[selected].weights.loc[VALIDATION[0] : VALIDATION[1]]
        contribution = (
            development_weights * panels["returns"].loc[development_weights.index]
        ).sum()
        removals = {
            "remove_top_5pct_contributors": set(
                contribution.nlargest(max(1, int(np.ceil(len(panels["symbols"]) * 0.05)))).index
            ),
            "remove_deterministic_20pct": {
                symbol for symbol in panels["symbols"]
                if int(hashlib.sha256(symbol.encode()).hexdigest()[:8], 16) % 5 == 0
            },
        }
        tests = [
            ("double_cost", selected_score, base, panels, replace(PortfolioCosts(), multiplier=2.0)),
            ("signal_delay_1", selected_score.shift(1), base, panels, PortfolioCosts()),
            ("rebalance_2", selected_score, replace(base, rebalance_every=2), panels, PortfolioCosts()),
            ("rebalance_5", selected_score, replace(base, rebalance_every=5), panels, PortfolioCosts()),
        ]
        if selected in filtered:
            forecast = artifacts["forecast_two"] if selected == "K08" else artifacts["forecast_four"]
            hurdle = artifacts["hurdle_two"] if selected == "K08" else artifacts["hurdle_four"]
            unfiltered = artifacts["two_equal"] if selected == "K08" else artifacts["four_icir"]
            base_multiplier = 1.0 if selected in {"K05", "K08"} else (1.5 if selected == "K06" else 2.0)
            for shift in (0.8, 1.2):
                perturbed, _ = apply_confidence_filter(
                    unfiltered, forecast, hurdle, base_multiplier * shift
                )
                tests.append(
                    (f"confidence_{shift:.1f}x", perturbed, base, panels, PortfolioCosts())
                )
        else:
            kill_rows.extend(
                [
                    {"test": "confidence_0.8x", "passed": np.nan, "status": "NOT_APPLICABLE"},
                    {"test": "confidence_1.2x", "passed": np.nan, "status": "NOT_APPLICABLE"},
                ]
            )
        for test_name, removed in removals.items():
            kept = [symbol for symbol in panels["symbols"] if symbol not in removed]
            tests.append(
                (test_name, selected_score[kept], base, subset_panels(panels, kept), PortfolioCosts())
            )
        for test_name, score, config, test_panels, costs in tests:
            result = run_candidate(
                score, config, test_panels, stress_artifact, costs, VALIDATION[1]
            )
            metrics = segment_metrics(result, test_panels["beta"], *VALIDATION)
            passed = result.status == "COMPLETED" and (
                metrics["sharpe"] > 0 if test_name == "double_cost" else metrics["total_return"] > 0
            )
            kill_rows.append({"test": test_name, "passed": passed, **metrics})
        kill_table = pd.DataFrame(kill_rows)
        kill_table.to_csv(output / "kill_tests.csv", index=False)
        applicable = kill_table.loc[kill_table.status.ne("NOT_APPLICABLE"), "passed"]
        kill_lock = {
            "status": "PASSED" if bool(applicable.astype(bool).all()) else "REJECTED",
            "candidate": selected,
            "kill_tests_sha256": sha256(output / "kill_tests.csv"),
        }
        runs[selected].daily.to_csv(output / "selected_daily.csv")
        runs[selected].weights.to_csv(output / "selected_weights.csv.gz", compression="gzip")
        runs[selected].rebalances.to_csv(output / "selected_rebalances.csv")
    write_json(output / "kill_test_lock.json", kill_lock)

    final_status = (
        "DEVELOPMENT_ACCEPTED_PENDING_PROSPECTIVE"
        if selected and kill_lock["status"] == "PASSED"
        else "REJECTED"
    )
    summary = {
        "status": final_status,
        "protocol_sha256": sha256(PROTOCOL),
        "selection": selection_lock,
        "kill_tests": kill_lock,
        "reused_audit": "NOT_LOADED_FOR_SELECTION_OR_KILL_TESTS",
        "prospective_start": "2026-09-25",
        "orders_allowed": False,
    }
    write_json(output / "SUMMARY.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("output/equities_2026-09-20_sip_r2"))
    parser.add_argument("--v2-output", type=Path, default=Path("reports/equity_v2/backtest"))
    parser.add_argument("--output", type=Path, default=Path("reports/equity_v9"))
    args = parser.parse_args()
    main(args.source, args.v2_output, args.output)

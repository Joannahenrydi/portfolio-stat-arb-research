"""Evaluate the frozen v10 expanded-universe, tail-selective experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd

from backtest.residual_reversal import ResidualReversalConfig, fit_stress_artifact
from features.adaptive_alpha import adaptive_kalman_innovation, remove_sector_peer_mean
from features.alphas import AlphaConfig, build_equity_features
from portfolio.optimizer import PortfolioCosts
from scripts.evaluate_equity_v2 import TRAIN, VALIDATION, read_panel
from scripts.evaluate_equity_v7 import (
    load_panels,
    run_candidate,
    segment_metrics as base_segment_metrics,
    sha256,
    subset_panels,
)
from scripts.evaluate_equity_v9 import standardize

PROTOCOL = Path("docs/EQUITY_RESEARCH_PROTOCOL_V10.md")


def tail_select(
    score: pd.DataFrame, eligibility: pd.DataFrame, fraction: float | None
) -> pd.DataFrame:
    if not score.index.equals(eligibility.index) or not score.columns.equals(eligibility.columns):
        raise ValueError("score and eligibility must align")
    usable = score.where(eligibility)
    if fraction is None:
        return usable
    if not np.isfinite(fraction) or not 0 < fraction < 0.5:
        raise ValueError("tail fraction must lie between zero and one half")
    percentile = usable.rank(axis=1, pct=True, method="average")
    tails = percentile.le(fraction) | percentile.gt(1 - fraction)
    return usable.where(tails)


def load_expanded_panels(source: Path, end: pd.Timestamp) -> tuple[dict, pd.DataFrame]:
    candidates = pd.read_csv(source / "equity_candidates.csv").sort_values("security_id")
    symbols = candidates.symbol.tolist()
    sectors = candidates.set_index("symbol").sector.reindex(symbols)
    calendar = pd.read_csv(source / "market_calendar.csv")
    sessions = pd.DatetimeIndex(pd.to_datetime(calendar.date))
    sessions = sessions[(sessions >= "2017-01-03") & (sessions <= end)]
    adjusted = read_panel(source / "bars_all.csv", symbols + ["SPY"], sessions, "c")
    raw_close = read_panel(source / "bars_raw.csv", symbols, sessions, "c")
    raw_volume = read_panel(source / "bars_raw.csv", symbols, sessions, "v")
    returns = adjusted[symbols].pct_change(fill_method=None)
    market = adjusted.SPY.pct_change(fill_method=None)
    sector_panel = pd.DataFrame(
        np.tile(sectors.to_numpy(), (len(sessions), 1)), index=sessions, columns=symbols
    )
    observed_history = returns.notna().rolling(252, min_periods=1).sum().shift(1)
    dollar_volume = raw_close * raw_volume
    adv = dollar_volume.rolling(60, min_periods=60).median().shift(1)
    liquidity_rank = adv.rank(axis=1, method="min", ascending=False)
    base = (
        returns.notna()
        & raw_close.ge(5)
        & raw_volume.gt(0)
        & observed_history.ge(252)
        & adv.ge(10_000_000)
        & liquidity_rank.le(400)
        & sector_panel.notna()
    )
    features = build_equity_features(
        returns, market, raw_volume, sector_panel, base, AlphaConfig()
    )
    residual_volatility = features.residual.rolling(60, min_periods=60).std().shift(1)
    eligibility = base & features.market_beta_prior.notna() & residual_volatility.notna()
    adaptive = adaptive_kalman_innovation(returns, market, mode="r_adaptive")
    score = standardize(
        remove_sector_peer_mean(adaptive.signal, sectors, eligibility), eligibility
    )
    prior_market_volatility = market.rolling(20, min_periods=20).std().shift(1) * np.sqrt(252)
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
        "prior_dispersion": features.residual.std(axis=1).shift(1),
        "prior_drawdown": adjusted.SPY.div(adjusted.SPY.cummax()).sub(1).shift(1),
        "scores": {},
        "symbols": symbols,
        "adaptive_score": score,
    }
    first_eligible = eligibility.apply(lambda column: column.index[column].min() if column.any() else pd.NaT)
    audit = candidates[["symbol", "security_id", "sector"]].copy()
    audit["first_eligible"] = audit.symbol.map(first_eligible)
    audit["train_eligible_sessions"] = audit.symbol.map(eligibility.loc[TRAIN[0] : TRAIN[1]].sum())
    audit["development_eligible_sessions"] = audit.symbol.map(
        eligibility.loc[VALIDATION[0] : VALIDATION[1]].sum()
    )
    return panels, audit


def segment_metrics(result, beta, start, end) -> dict:
    metrics = base_segment_metrics(result, beta, start, end)
    if metrics.get("status") == "COMPLETED":
        rebalances = result.rebalances.loc[start:end]
        metrics["average_eligible_names"] = float(rebalances.eligible_names.mean())
        metrics["neutrality_failures"] = 0
    return metrics


def passes_gate(train: dict, development: dict, tolerance: float) -> bool:
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


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False, default=str) + "\n")


def main(source: Path, v2_output: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    legacy, legacy_audit = load_panels(source, v2_output, VALIDATION[1])
    expanded, expanded_audit = load_expanded_panels(source, VALIDATION[1])
    expanded_audit.to_csv(output / "expanded_universe_audit.csv", index=False)
    legacy_audit.to_csv(output / "legacy_universe_audit.csv", index=False)
    legacy_adaptive = adaptive_kalman_innovation(
        legacy["returns"], legacy["market"], mode="r_adaptive"
    )
    legacy_score = standardize(
        remove_sector_peer_mean(
            legacy_adaptive.signal, legacy["sectors"], legacy["eligibility"]
        ),
        legacy["eligibility"],
    )
    raw = expanded["adaptive_score"]
    smooth = raw.ewm(halflife=3, adjust=False, min_periods=1).mean().where(
        expanded["eligibility"]
    )
    matrix = {
        "X00": (legacy_score, legacy, None, 3, False),
        "X01": (tail_select(raw, expanded["eligibility"], None), expanded, None, 3, False),
        "X02": (tail_select(raw, expanded["eligibility"], 0.25), expanded, 0.25, 3, False),
        "X03": (tail_select(raw, expanded["eligibility"], 0.15), expanded, 0.15, 3, False),
        "X04": (tail_select(raw, expanded["eligibility"], 0.25), expanded, 0.25, 5, False),
        "X05": (tail_select(raw, expanded["eligibility"], 0.15), expanded, 0.15, 5, False),
        "X06": (tail_select(raw, expanded["eligibility"], 0.25), expanded, 0.25, 10, False),
        "X07": (tail_select(raw, expanded["eligibility"], 0.15), expanded, 0.15, 10, False),
        "X08": (tail_select(smooth, expanded["eligibility"], 0.15), expanded, 0.15, 5, True),
    }
    freeze = {
        "protocol_sha256": sha256(PROTOCOL),
        "source_hashes": {
            "adjusted_bars": sha256(source / "bars_all.csv"),
            "raw_bars": sha256(source / "bars_raw.csv"),
            "candidate_master": sha256(source / "equity_candidates.csv"),
        },
        "matrix": {
            name: {"tail_fraction": tail, "rebalance_every": rebalance, "ewma3": ewma}
            for name, (_, _, tail, rebalance, ewma) in matrix.items()
        },
        "costs": asdict(PortfolioCosts()),
        "reused_audit_loaded_for_selection": False,
        "expanded_source_names": len(expanded["symbols"]),
        "data_label": "research_snapshot_only",
    }
    write_json(output / "frozen_run_inputs.json", freeze)

    costs = PortfolioCosts()
    runs, rows = {}, []
    for name, (score, panels, _, rebalance, _) in matrix.items():
        config = ResidualReversalConfig(rebalance_every=rebalance)
        stress = fit_stress_artifact(panels["prior_market_volatility"], *TRAIN)
        result = run_candidate(score, config, panels, stress, costs, VALIDATION[1])
        runs[name] = result
        train = segment_metrics(result, panels["beta"], *TRAIN)
        development = segment_metrics(result, panels["beta"], *VALIDATION)
        row = {
            "experiment": name,
            "status": "REFERENCE_ONLY" if name == "X00" and result.status == "COMPLETED" else result.status,
            "reason": result.reason,
            "eligible": name != "X00" and passes_gate(train, development, config.neutrality_tolerance),
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
    selectable = selection.loc[selection.experiment.ne("X00")].sort_values(
        ["eligible", "robust_sharpe", "worst_calendar_year", "development_cagr",
         "development_annual_turnover", "experiment"],
        ascending=[False, False, False, False, True, True], na_position="last",
    )
    selection = pd.concat(
        [selection.loc[selection.experiment.eq("X00")], selectable], ignore_index=True
    )
    selection.to_csv(output / "candidate_selection.csv", index=False)
    selected = str(selectable.loc[selectable.eligible].iloc[0].experiment) if selectable.eligible.any() else None
    selection_lock = {
        "status": "SELECTED_PENDING_KILL_TESTS" if selected else "REJECTED",
        "reason": None if selected else "NO_V10_CANDIDATE_PASSED_FROZEN_ACCEPTANCE_GATES",
        "candidate": selected,
        "selection_sha256": sha256(output / "candidate_selection.csv"),
        "reused_audit_evaluated": False,
    }
    write_json(output / "selection_lock.json", selection_lock)

    kill_rows = []
    if selected is None:
        pd.DataFrame(columns=["test", "passed", "status"]).to_csv(
            output / "kill_tests.csv", index=False
        )
        kill_lock = {"status": "NOT_RUN_NO_SELECTED_CANDIDATE"}
    else:
        score, panels, fraction, rebalance, ewma = matrix[selected]
        config = ResidualReversalConfig(rebalance_every=rebalance)
        stress = fit_stress_artifact(panels["prior_market_volatility"], *TRAIN)
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
            ("double_cost", score, config, panels, replace(costs, multiplier=2.0)),
            ("signal_delay_1", score.shift(1), config, panels, costs),
        ]
        if fraction is not None:
            base_signal = smooth if ewma else raw
            for scale in (0.8, 1.2):
                perturbed = tail_select(base_signal, expanded["eligibility"], fraction * scale)
                tests.append((f"tail_{scale:.1f}x", perturbed, config, expanded, costs))
        for neighboring in sorted({max(1, rebalance - 2), rebalance + 2}):
            tests.append(
                (f"rebalance_{neighboring}", score,
                 replace(config, rebalance_every=neighboring), panels, costs)
            )
        for test_name, removed in removals.items():
            kept = [symbol for symbol in panels["symbols"] if symbol not in removed]
            tests.append(
                (test_name, score[kept], config, subset_panels(panels, kept), costs)
            )
        for test_name, test_score, test_config, test_panels, test_costs in tests:
            result = run_candidate(
                test_score, test_config, test_panels, stress, test_costs, VALIDATION[1]
            )
            metrics = segment_metrics(result, test_panels["beta"], *VALIDATION)
            passed = result.status == "COMPLETED" and (
                metrics["sharpe"] > 0 if test_name == "double_cost" else metrics["total_return"] > 0
            )
            kill_rows.append({"test": test_name, "passed": passed, **metrics})
        kill_table = pd.DataFrame(kill_rows)
        kill_table.to_csv(output / "kill_tests.csv", index=False)
        kill_lock = {
            "status": "PASSED" if bool(kill_table.passed.all()) else "REJECTED",
            "candidate": selected,
            "kill_tests_sha256": sha256(output / "kill_tests.csv"),
        }
        runs[selected].daily.to_csv(output / "selected_daily.csv")
        runs[selected].weights.to_csv(output / "selected_weights.csv.gz", compression="gzip")
        runs[selected].rebalances.to_csv(output / "selected_rebalances.csv")
    write_json(output / "kill_test_lock.json", kill_lock)
    status = (
        "DEVELOPMENT_ACCEPTED_PENDING_REUSED_AUDIT_AND_PROSPECTIVE"
        if selected and kill_lock["status"] == "PASSED"
        else "REJECTED"
    )
    write_json(
        output / "SUMMARY.json",
        {"status": status, "selection": selection_lock, "kill_tests": kill_lock,
         "reused_audit": "NOT_LOADED", "orders_allowed": False},
    )
    print(json.dumps({"status": status, "selection": selection_lock, "kill_tests": kill_lock}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("output/equities_2026-09-20_sip_r2"))
    parser.add_argument("--v2-output", type=Path, default=Path("reports/equity_v2/backtest"))
    parser.add_argument("--output", type=Path, default=Path("reports/equity_v10"))
    args = parser.parse_args()
    main(args.source, args.v2_output, args.output)

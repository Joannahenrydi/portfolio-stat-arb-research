"""Run the frozen v11 train-qualified OHLCV alpha discovery matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd

from backtest.residual_reversal import ResidualReversalConfig, fit_stress_artifact
from features.adaptive_alpha import remove_sector_peer_mean
from features.alphas import forward_total_return_labels
from portfolio.optimizer import PortfolioCosts
from scripts.evaluate_equity_v2 import TRAIN, VALIDATION, read_panel
from scripts.evaluate_equity_v7 import run_candidate, sha256, subset_panels
from scripts.evaluate_equity_v9 import standardize
from scripts.evaluate_equity_v10 import (
    load_expanded_panels,
    passes_gate,
    segment_metrics,
    tail_select,
    write_json,
)

PROTOCOL = Path("docs/EQUITY_RESEARCH_PROTOCOL_V11.md")
HORIZONS = {
    "reversal_1": 5,
    "reversal_3": 5,
    "reversal_5": 5,
    "momentum_21_5": 10,
    "momentum_63_5": 10,
    "momentum_126_21": 21,
    "momentum_252_21": 21,
    "gap_reversal": 5,
    "intraday_reversal": 5,
    "close_location": 5,
    "volume_shock_reversal": 5,
    "high_252_momentum": 21,
}


def internal_gap_lengths(volume: pd.DataFrame) -> pd.Series:
    """Largest missing/nonpositive run strictly between first and last valid bar."""
    output = {}
    for symbol in volume:
        valid = volume[symbol].notna() & volume[symbol].gt(0)
        locations = np.flatnonzero(valid.to_numpy())
        if len(locations) < 2:
            output[symbol] = 0
            continue
        inside = ~valid.iloc[locations[0] : locations[-1] + 1]
        groups = inside.ne(inside.shift()).cumsum()
        output[symbol] = int(inside.groupby(groups).sum().max())
    return pd.Series(output, dtype=int)


def identity_gap_exclusions(source: Path) -> tuple[set[str], pd.DataFrame]:
    candidates = pd.read_csv(source / "equity_candidates.csv").sort_values("security_id")
    calendar = pd.DatetimeIndex(pd.to_datetime(pd.read_csv(source / "market_calendar.csv").date))
    volume = read_panel(source / "bars_raw.csv", candidates.symbol.tolist(), calendar, "v")
    gaps = internal_gap_lengths(volume)
    audit = candidates[["symbol", "security_id", "sector"]].copy()
    audit["maximum_internal_missing_or_zero_run"] = audit.symbol.map(gaps)
    audit["excluded"] = audit.maximum_internal_missing_or_zero_run.ge(20)
    return set(audit.loc[audit.excluded, "symbol"]), audit


def feature_library(source: Path, panels: dict) -> dict[str, pd.DataFrame]:
    symbols, sessions = panels["symbols"], panels["returns"].index
    adjusted_open = read_panel(source / "bars_all.csv", symbols, sessions, "o")
    adjusted_high = read_panel(source / "bars_all.csv", symbols, sessions, "h")
    adjusted_low = read_panel(source / "bars_all.csv", symbols, sessions, "l")
    adjusted_close = read_panel(source / "bars_all.csv", symbols, sessions, "c")
    residual = panels["residual"]
    log_volume = np.log1p(panels["raw_volume"])
    volume_surprise = log_volume - log_volume.rolling(60, min_periods=60).median().shift(1)
    gap = adjusted_open.div(adjusted_close.shift(1)).sub(1)
    intraday = adjusted_close.div(adjusted_open).sub(1)
    spread = adjusted_high.sub(adjusted_low)
    close_location = adjusted_close.sub(adjusted_low).div(spread.where(spread.gt(0))).sub(0.5)
    rolling_high = adjusted_high.rolling(252, min_periods=252).max()
    raw = {
        "reversal_1": -residual,
        "reversal_3": -residual.rolling(3, min_periods=3).sum(),
        "reversal_5": -residual.rolling(5, min_periods=5).sum(),
        "momentum_21_5": residual.rolling(21, min_periods=21).sum()
        - residual.rolling(5, min_periods=5).sum(),
        "momentum_63_5": residual.rolling(63, min_periods=63).sum()
        - residual.rolling(5, min_periods=5).sum(),
        "momentum_126_21": residual.rolling(126, min_periods=126).sum()
        - residual.rolling(21, min_periods=21).sum(),
        "momentum_252_21": residual.rolling(252, min_periods=252).sum()
        - residual.rolling(21, min_periods=21).sum(),
        "gap_reversal": -gap,
        "intraday_reversal": -intraday,
        "close_location": close_location,
        "volume_shock_reversal": -residual * volume_surprise.abs(),
        "high_252_momentum": adjusted_close.div(rolling_high.where(rolling_high.gt(0))),
    }
    return {
        name: standardize(
            remove_sector_peer_mean(frame, panels["sectors"], panels["eligibility"]),
            panels["eligibility"],
        )
        for name, frame in raw.items()
    }


def qualify_features(
    features: dict[str, pd.DataFrame], returns: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    labels = {}
    ends = {}
    for horizon in sorted(set(HORIZONS.values())):
        labels[horizon], ends[horizon] = forward_total_return_labels(returns, horizon)
    rows = []
    daily_ics = {}
    for name in HORIZONS:
        horizon = HORIZONS[name]
        completed = ends[horizon].le(TRAIN[1])
        signal = features[name].where(completed).loc[TRAIN[0] : TRAIN[1]]
        target = labels[horizon].where(completed).loc[TRAIN[0] : TRAIN[1]]
        daily = signal.corrwith(target, axis=1, method="spearman")
        daily_ics[name] = daily
        annual = daily.groupby(daily.index.year).mean()
        mean_ic = float(daily.mean())
        std_ic = float(daily.std(ddof=1))
        icir = mean_ic / std_ic if std_ic > 0 else np.nan
        positive_years = int(annual.reindex(range(2018, 2023)).gt(0).sum())
        rows.append(
            {"family": name, "horizon": horizon, "mean_rank_ic": mean_ic,
             "ic_std": std_ic, "icir": icir, "positive_years": positive_years,
             "qualified_raw": bool(mean_ic > 0 and icir >= 0.02 and positive_years >= 4)}
        )
    qualification = pd.DataFrame(rows).set_index("family")
    ic_frame = pd.DataFrame(daily_ics)
    admitted = []
    ordered = qualification.loc[qualification.qualified_raw].sort_values(
        ["icir", "mean_rank_ic"], ascending=False
    ).index
    for name in ordered:
        if all(abs(ic_frame[name].corr(ic_frame[prior])) < 0.75 for prior in admitted):
            admitted.append(name)
        if len(admitted) == 4:
            break
    qualification["admitted"] = qualification.index.isin(admitted)
    return qualification.reset_index(), ic_frame.corr(), admitted


def blend(features: dict[str, pd.DataFrame], names: list[str], weights=None) -> pd.DataFrame:
    if not names:
        raise ValueError("blend needs at least one family")
    if weights is None:
        weights = np.full(len(names), 1 / len(names))
    weights = np.asarray(weights, dtype=float)
    weights = weights / weights.sum()
    return sum(weight * features[name] for name, weight in zip(names, weights))


def main(source: Path, v2_output: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    exclusions, identity_audit = identity_gap_exclusions(source)
    identity_audit.to_csv(output / "identity_integrity_audit.csv", index=False)
    dirty, _ = load_expanded_panels(source, VALIDATION[1])
    panels, universe_audit = load_expanded_panels(source, VALIDATION[1], exclusions)
    universe_audit.to_csv(output / "expanded_clean_universe_audit.csv", index=False)
    features = feature_library(source, panels)
    qualification, ic_correlation, admitted = qualify_features(features, panels["returns"])
    qualification.to_csv(output / "train_feature_qualification.csv", index=False)
    ic_correlation.to_csv(output / "train_daily_ic_correlation.csv")
    qualified = qualification.set_index("family")
    top = admitted[:1]
    top_two = admitted[:2]
    top_four = admitted
    momentum = [name for name in admitted if "momentum" in name]
    reversal = [name for name in admitted if "reversal" in name]

    def icir_weights(names):
        values = qualified.loc[names, "icir"].clip(lower=0).to_numpy(dtype=float)
        return values / values.sum()

    candidate_scores = {}
    blocked = {}
    if top:
        candidate_scores["A01"] = blend(features, top)
    else:
        blocked["A01"] = "NO_QUALIFIED_FAMILY"
    if len(top_two) >= 2:
        candidate_scores["A02"] = blend(features, top_two)
    else:
        blocked["A02"] = "FEWER_THAN_TWO_ADMITTED_FAMILIES"
    if len(top_four) >= 2:
        candidate_scores["A03"] = blend(features, top_four, icir_weights(top_four))
        candidate_scores["A04"] = candidate_scores["A03"]
        candidate_scores["A07"] = candidate_scores["A03"]
        candidate_scores["A08"] = candidate_scores["A03"]
    else:
        for name in ("A03", "A04", "A07", "A08"):
            blocked[name] = "FEWER_THAN_TWO_ADMITTED_FAMILIES"
    if momentum:
        candidate_scores["A05"] = blend(features, momentum, icir_weights(momentum))
    else:
        blocked["A05"] = "NO_ADMITTED_MOMENTUM_FAMILY"
    if reversal:
        candidate_scores["A06"] = blend(features, reversal, icir_weights(reversal))
    else:
        blocked["A06"] = "NO_ADMITTED_REVERSAL_FAMILY"

    matrix_meta = {
        "A00": (0.25, ResidualReversalConfig(rebalance_every=5), dirty),
        "A01": (0.25, ResidualReversalConfig(rebalance_every=10), panels),
        "A02": (0.25, ResidualReversalConfig(rebalance_every=10), panels),
        "A03": (0.25, ResidualReversalConfig(rebalance_every=10), panels),
        "A04": (0.25, ResidualReversalConfig(rebalance_every=21), panels),
        "A05": (0.25, ResidualReversalConfig(rebalance_every=21), panels),
        "A06": (0.25, ResidualReversalConfig(rebalance_every=5), panels),
        "A07": (
            0.25,
            ResidualReversalConfig(
                rebalance_every=21, max_gross=2.0, max_name=0.02,
                max_turnover=0.50, target_buffer=0.95,
            ),
            panels,
        ),
        "A08": (0.15, ResidualReversalConfig(rebalance_every=10), panels),
    }
    candidate_scores["A00"] = dirty["adaptive_score"]
    freeze = {
        "protocol_sha256": sha256(PROTOCOL),
        "excluded_identity_gap_symbols": sorted(exclusions),
        "admitted_families": admitted,
        "candidate_configs": {
            name: {"tail_fraction": tail, **asdict(config)}
            for name, (tail, config, _) in matrix_meta.items()
        },
        "blocked_candidates": blocked,
        "reused_audit_loaded": False,
    }
    write_json(output / "frozen_run_inputs.json", freeze)

    runs, rows = {}, []
    for name in matrix_meta:
        if name in blocked:
            rows.append(
                {"experiment": name, "status": "BLOCKED", "reason": blocked[name],
                 "eligible": False}
            )
            continue
        fraction, config, test_panels = matrix_meta[name]
        score = tail_select(
            candidate_scores[name], test_panels["eligibility"], fraction
        )
        stress = fit_stress_artifact(test_panels["prior_market_volatility"], *TRAIN)
        result = run_candidate(
            score, config, test_panels, stress, PortfolioCosts(), VALIDATION[1]
        )
        runs[name] = result
        train = segment_metrics(result, test_panels["beta"], *TRAIN)
        development = segment_metrics(result, test_panels["beta"], *VALIDATION)
        row = {
            "experiment": name,
            "status": "REFERENCE_ONLY" if name == "A00" and result.status == "COMPLETED" else result.status,
            "reason": result.reason,
            "eligible": name != "A00" and passes_gate(
                train, development, config.neutrality_tolerance
            ),
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
    selectable = selection.loc[selection.experiment.ne("A00")].sort_values(
        ["eligible", "robust_sharpe", "worst_calendar_year", "development_cagr",
         "development_annual_turnover", "experiment"],
        ascending=[False, False, False, False, True, True], na_position="last",
    )
    selection = pd.concat(
        [selection.loc[selection.experiment.eq("A00")], selectable], ignore_index=True
    )
    selection.to_csv(output / "candidate_selection.csv", index=False)
    selected = str(selectable.loc[selectable.eligible].iloc[0].experiment) if selectable.eligible.any() else None
    selection_lock = {
        "status": "SELECTED_PENDING_KILL_TESTS" if selected else "REJECTED",
        "reason": None if selected else "NO_V11_CANDIDATE_PASSED_FROZEN_ACCEPTANCE_GATES",
        "candidate": selected,
        "selection_sha256": sha256(output / "candidate_selection.csv"),
        "reused_audit_evaluated": False,
    }
    write_json(output / "selection_lock.json", selection_lock)

    if selected is None:
        pd.DataFrame(columns=["test", "passed", "status"]).to_csv(
            output / "kill_tests.csv", index=False
        )
        kill_lock = {"status": "NOT_RUN_NO_SELECTED_CANDIDATE"}
    else:
        fraction, config, selected_panels = matrix_meta[selected]
        base_score = candidate_scores[selected]
        score = tail_select(base_score, selected_panels["eligibility"], fraction)
        stress = fit_stress_artifact(selected_panels["prior_market_volatility"], *TRAIN)
        contribution = (
            runs[selected].weights.loc[VALIDATION[0] : VALIDATION[1]]
            * selected_panels["returns"].loc[VALIDATION[0] : VALIDATION[1]]
        ).sum()
        removals = {
            "remove_top_5pct_contributors": set(
                contribution.nlargest(max(1, int(np.ceil(len(selected_panels["symbols"]) * 0.05)))).index
            ),
            "remove_deterministic_20pct": {
                symbol for symbol in selected_panels["symbols"]
                if int(hashlib.sha256(symbol.encode()).hexdigest()[:8], 16) % 5 == 0
            },
        }
        tests = [
            ("double_cost", score, config, selected_panels, replace(PortfolioCosts(), multiplier=2.0)),
            ("signal_delay_1", score.shift(1), config, selected_panels, PortfolioCosts()),
        ]
        for scale in (0.8, 1.2):
            tests.append(
                (f"tail_{scale:.1f}x", tail_select(base_score, selected_panels["eligibility"], fraction * scale),
                 config, selected_panels, PortfolioCosts())
            )
        for interval in sorted({max(1, config.rebalance_every - 3), config.rebalance_every + 3}):
            tests.append(
                (f"rebalance_{interval}", score, replace(config, rebalance_every=interval),
                 selected_panels, PortfolioCosts())
            )
        for test_name, removed in removals.items():
            kept = [symbol for symbol in selected_panels["symbols"] if symbol not in removed]
            tests.append(
                (test_name, score[kept], config, subset_panels(selected_panels, kept), PortfolioCosts())
            )
        kill_rows = []
        for test_name, test_score, test_config, test_panels, costs in tests:
            result = run_candidate(
                test_score, test_config, test_panels, stress, costs, VALIDATION[1]
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
    status = "DEVELOPMENT_ACCEPTED_PENDING_PROSPECTIVE" if selected and kill_lock["status"] == "PASSED" else "REJECTED"
    write_json(
        output / "SUMMARY.json",
        {"status": status, "selection": selection_lock, "kill_tests": kill_lock,
         "reused_audit": "NOT_LOADED", "orders_allowed": False},
    )
    print(json.dumps({"status": status, "admitted": admitted, "selection": selection_lock,
                      "kill_tests": kill_lock}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("output/equities_2026-09-20_sip_r2"))
    parser.add_argument("--v2-output", type=Path, default=Path("reports/equity_v2/backtest"))
    parser.add_argument("--output", type=Path, default=Path("reports/equity_v11"))
    args = parser.parse_args()
    main(args.source, args.v2_output, args.output)

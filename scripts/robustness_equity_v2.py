"""Run declared kill tests for the locked v2 candidate."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from evaluate_equity_v2 import AUDIT, TRAIN, VALIDATION, read_panel, segment_metrics

from backtest.engine import (
    FastPortfolioConfig,
    candidate_scores,
    residual_returns,
    run_fast_backtest,
)
from portfolio.optimizer import PortfolioCosts


def main() -> None:
    source = Path("output/equities_2026-09-20_sip_r2")
    v2 = Path("reports/equity_v2/backtest")
    output = Path("reports/equity_v2/robustness")
    output.mkdir(parents=True, exist_ok=True)
    lock = json.loads((v2 / "selection_lock.json").read_text())
    selected = lock["selected_candidate"]
    initial = pd.read_csv("reports/equity_v2/data_quality/prospective_400_stocks.csv")
    universe_audit = pd.read_csv(v2 / "backtest_universe_audit.csv")
    symbols = universe_audit.loc[universe_audit.included, "symbol"].tolist()
    universe = initial[initial.symbol.isin(symbols)]
    sectors = universe.set_index("symbol").sector.reindex(symbols)
    calendar = pd.read_csv(source / "market_calendar.csv")
    sessions = pd.DatetimeIndex(pd.to_datetime(calendar.date))
    sessions = sessions[(sessions >= "2017-01-03") & (sessions <= AUDIT[1])]
    close = read_panel(source / "bars_all.csv", symbols + ["SPY"], sessions, "c")
    raw_close = read_panel(source / "bars_raw.csv", symbols, sessions, "c")
    raw_volume = read_panel(source / "bars_raw.csv", symbols, sessions, "v")
    returns = close[symbols].pct_change(fill_method=None)
    market = close.SPY.pct_change(fill_method=None)
    residual, beta = residual_returns(returns, market, sectors)
    score = candidate_scores(residual)[selected]
    study = returns.index >= TRAIN[0]
    frames = (returns.loc[study], beta.loc[study], raw_close.loc[study], raw_volume.loc[study])
    base_config = FastPortfolioConfig()
    base_costs = PortfolioCosts()

    scenarios: list[tuple[str, pd.DataFrame, FastPortfolioConfig, PortfolioCosts]] = [
        ("base", score.loc[study], base_config, base_costs),
        ("costs_x2", score.loc[study], base_config, replace(base_costs, multiplier=2.0)),
        ("signal_delay_1", score.shift(1).loc[study], base_config, base_costs),
        ("rebalance_10", score.loc[study], replace(base_config, rebalance_every=10), base_costs),
        ("rebalance_21", score.loc[study], replace(base_config, rebalance_every=21), base_costs),
    ]
    # Deterministic 20% universe perturbation fixed by security ID hash.
    dropped = universe.loc[
        universe.security_id.map(lambda value: int(hashlib.sha256(value.encode()).hexdigest(), 16) % 5 == 0),
        "symbol",
    ].tolist()
    perturbed = score.copy()
    perturbed.loc[:, dropped] = np.nan
    scenarios.append(("drop_deterministic_20pct", perturbed.loc[study], base_config, base_costs))

    rows = []
    base_weights = None
    for scenario, scenario_score, config, costs in scenarios:
        daily, weights = run_fast_backtest(
            scenario_score, frames[0], frames[1], sectors, frames[2], frames[3],
            config=config, costs=costs,
        )
        if scenario == "base":
            base_weights = weights
        for segment, bounds in (("train", TRAIN), ("validation", VALIDATION), ("reused_audit", AUDIT)):
            rows.append({"scenario": scenario, "segment": segment,
                         **segment_metrics(daily, weights, beta, *bounds)})
    # Remove the top 5% validation gross contributors, then keep that list fixed.
    validation_index = base_weights.loc[VALIDATION[0]:VALIDATION[1]].index
    contributions = (base_weights.loc[validation_index] * returns.loc[validation_index]).sum()
    count = max(1, int(len(symbols) * 0.05))
    best = contributions.nlargest(count).index.tolist()
    (output / "removed_best_5pct_symbols.json").write_text(json.dumps(best, indent=2) + "\n")
    removed = score.copy()
    removed.loc[:, best] = np.nan
    daily, weights = run_fast_backtest(
        removed.loc[study], frames[0], frames[1], sectors, frames[2], frames[3],
        config=base_config, costs=base_costs,
    )
    for segment, bounds in (("train", TRAIN), ("validation", VALIDATION), ("reused_audit", AUDIT)):
        rows.append({"scenario": "remove_validation_best_5pct", "segment": segment,
                     **segment_metrics(daily, weights, beta, *bounds)})
    result = pd.DataFrame(rows)
    result.to_csv(output / "kill_tests.csv", index=False)
    summary = {
        "selected_candidate": selected,
        "scenarios": sorted(result.scenario.unique()),
        "base_passed_original_development_gate": False,
        "robustness_decision": "REJECTED",
        "reason": "base strategy has nonpositive training Sharpe and weak validation Sharpe",
    }
    (output / "decision.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(result[result.segment.ne("train")][
        ["scenario", "segment", "cagr", "sharpe", "max_drawdown", "annual_turnover"]
    ].to_string(index=False))


if __name__ == "__main__":
    main()

"""Build one costed covariance-aware target from the locked equity candidate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from backtest.engine import candidate_scores, residual_returns
from features.alphas import (
    estimate_shrinkage_covariance,
    fit_expected_return_calibration,
    forward_total_return_labels,
    predict_expected_returns,
)
from portfolio.neutralization import build_exposure_matrix, exposure_report
from portfolio.optimizer import PortfolioConstraints, construct_portfolio
from scripts.evaluate_equity_v2 import TRAIN, read_panel


def main(source: Path, quality: Path, selection_path: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    universe = pd.read_csv(quality / "prospective_400_stocks.csv").sort_values("security_id")
    symbols = universe.symbol.tolist()
    sectors = universe.set_index("symbol").sector.reindex(symbols)
    sessions = pd.DatetimeIndex(pd.to_datetime(pd.read_csv(source / "market_calendar.csv").date))
    sessions = sessions[(sessions >= "2017-01-03") & (sessions <= "2026-09-18")]
    adjusted = read_panel(source / "bars_all.csv", symbols + ["SPY"], sessions, "c")
    study = sessions[(sessions >= TRAIN[0])]
    complete = adjusted.loc[study, symbols].notna().all()
    discontinuity = adjusted[symbols].pct_change(fill_method=None).loc[study].abs().gt(0.50).any()
    symbols = [symbol for symbol in symbols if complete[symbol] and not discontinuity[symbol]]
    sectors = sectors.reindex(symbols)
    adjusted = adjusted[symbols + ["SPY"]]
    raw_close = read_panel(source / "bars_raw.csv", symbols, sessions, "c")
    raw_volume = read_panel(source / "bars_raw.csv", symbols, sessions, "v")
    returns = adjusted[symbols].pct_change(fill_method=None)
    market = adjusted.SPY.pct_change(fill_method=None)
    residual, beta = residual_returns(returns, market, sectors)
    winner = json.loads(selection_path.read_text())["selected_candidate"]
    score = candidate_scores(residual)[winner]

    labels, label_end = forward_total_return_labels(returns, horizon=5)
    artifact = fit_expected_return_calibration(
        {"locked_score": score},
        labels,
        label_end,
        train_start=TRAIN[0],
        train_end=TRAIN[1],
        pit_verified=False,
        input_scope="research_snapshot_only",
        min_samples=100_000,
    )
    decision = score.index[-1]
    alpha = predict_expected_returns(
        {"locked_score": score.loc[[decision]]}, artifact
    ).iloc[0]
    covariance = estimate_shrinkage_covariance(
        returns[symbols], decision, lookback=252, min_observations=200, shrinkage=0.2
    ).covariance * 5
    exposures = build_exposure_matrix(beta.loc[decision], sectors)
    adv = (raw_close * raw_volume).rolling(60, min_periods=60).median().loc[decision]
    result = construct_portfolio(
        alpha,
        covariance,
        exposures,
        pd.Series(0.0, index=symbols),
        adv,
        1_000_000.0,
        constraints=PortfolioConstraints(max_turnover=1.0),
    )
    target = result.target.rename("theoretical_weight").rename_axis("symbol").reset_index()
    target = target.merge(universe[["symbol", "security_id", "sector"]], on="symbol", how="left")
    target["order_authorized"] = False
    target.to_csv(output / "optimized_target.csv", index=False)
    exposure = exposure_report(result.target, exposures)
    report = {
        "decision_session": str(decision.date()),
        "candidate": winner,
        "data_scope": "research_snapshot_only",
        "pit_verified": False,
        "calibration_train_end": str(artifact.train_end.date()),
        "calibration_samples": artifact.sample_count,
        "calibration_digest": artifact.training_digest,
        "status": result.status,
        "optimizer_metrics": result.metrics,
        "exposure": {name: float(value) for name, value in exposure.items()},
        "promotion_status": "REJECTED",
        "orders_allowed": False,
    }
    (output / "optimized_target.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("output/equities_2026-09-20_sip_r2"))
    parser.add_argument("--quality", type=Path, default=Path("reports/equity_v2/data_quality"))
    parser.add_argument(
        "--selection", type=Path, default=Path("reports/equity_v2/backtest/selection_lock.json")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("reports/equity_final/optimized_portfolio")
    )
    args = parser.parse_args()
    main(args.source, args.quality, args.selection, args.output)

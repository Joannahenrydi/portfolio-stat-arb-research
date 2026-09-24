"""Create mandatory diagnostics for the best rejected v9 research candidate."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from backtest.residual_reversal import ResidualReversalConfig, fit_stress_artifact
from portfolio.optimizer import PortfolioCosts
from scripts.diagnose_equity_v7 import holding_episodes
from scripts.evaluate_equity_v2 import TRAIN, VALIDATION
from scripts.evaluate_equity_v7 import load_panels, run_candidate
from scripts.evaluate_equity_v9 import build_v9_artifacts


def sharpe(values: pd.Series) -> float:
    std = values.std(ddof=1)
    return float(values.mean() / std * np.sqrt(252)) if std > 0 else np.nan


def main(source: Path, v2_output: Path, output: Path) -> None:
    selection = pd.read_csv(output / "candidate_selection.csv")
    diagnostic_pool = selection[
        selection.experiment.ne("K00")
        & selection.status.eq("COMPLETED")
        & selection.development_sharpe.notna()
    ]
    candidate = str(
        diagnostic_pool.sort_values("robust_sharpe", ascending=False).iloc[0].experiment
    )
    panels, _ = load_panels(source, v2_output, VALIDATION[1])
    artifacts = build_v9_artifacts(panels)
    score_map = {
        "K01": artifacts["r_signal"],
        "K02": artifacts["qr_signal"],
        "K03": artifacts["four_equal"],
        "K04": artifacts["four_icir"],
    }
    if candidate not in score_map:
        raise RuntimeError("best rejected diagnostic candidate must have nonempty positions")
    score = score_map[candidate]
    stress = fit_stress_artifact(panels["prior_market_volatility"], *TRAIN)
    result = run_candidate(
        score, ResidualReversalConfig(), panels, stress, PortfolioCosts(), VALIDATION[1]
    )
    if result.status != "COMPLETED":
        raise RuntimeError(result.reason)
    daily = result.daily.copy()
    weights = result.weights.copy()
    returns = panels["returns"].loc[daily.index]
    contributions = weights * returns
    daily["long_pnl"] = contributions.where(weights.gt(0), 0).sum(axis=1)
    daily["short_pnl"] = contributions.where(weights.lt(0), 0).sum(axis=1)
    daily["cost_drag"] = daily.transaction_cost + daily.borrow_cost
    daily.to_csv(output / "diagnostic_daily.csv")
    weights.to_csv(output / "diagnostic_weights.csv.gz", compression="gzip")

    complete = artifacts["label_end"].le(VALIDATION[1])
    labels = artifacts["labels"].where(complete)
    ic = pd.DataFrame(
        {
            "ic": score.where(complete).corrwith(labels, axis=1),
            "rank_ic": score.where(complete).corrwith(labels, axis=1, method="spearman"),
        }
    ).loc[TRAIN[0] : VALIDATION[1]]
    ic.groupby(ic.index.year).mean().reset_index(names="year").to_csv(
        output / "diagnostic_ic_by_year.csv", index=False
    )

    annual_rows = []
    for year, frame in daily.groupby(daily.index.year):
        annual_rows.append(
            {
                "year": int(year),
                "gross_return": float((1 + frame.gross_return).prod() - 1),
                "net_return": float((1 + frame.net_return).prod() - 1),
                "gross_sharpe": sharpe(frame.gross_return),
                "net_sharpe": sharpe(frame.net_return),
                "turnover": float(frame.turnover.sum()),
                "transaction_cost": float(frame.transaction_cost.sum()),
                "borrow_cost": float(frame.borrow_cost.sum()),
                "long_pnl": float(frame.long_pnl.sum()),
                "short_pnl": float(frame.short_pnl.sum()),
            }
        )
    pd.DataFrame(annual_rows).to_csv(output / "annual_attribution.csv", index=False)

    episodes = holding_episodes(weights)
    completed = episodes.loc[episodes.completed].copy()
    completed["year"] = pd.to_datetime(completed.entry).dt.year
    completed.groupby(["year", "side"]).sessions.agg(
        ["mean", "median", "count"]
    ).reset_index().to_csv(output / "holding_period_by_year.csv", index=False)

    percentile = panels["residual_volatility"].loc[daily.index].rank(axis=1, pct=True)
    bucket_rows = []
    for segment, (start, end) in (("train", TRAIN), ("development", VALIDATION)):
        years = len(daily.loc[start:end]) / 252
        for bucket in range(1, 6):
            mask = percentile.gt((bucket - 1) / 5) & percentile.le(bucket / 5)
            pnl = float(contributions.loc[start:end].where(mask.loc[start:end], 0).sum().sum())
            bucket_rows.append(
                {"segment": segment, "bucket": f"Q{bucket}",
                 "pnl_annualized_additive": pnl / years}
            )
    pd.DataFrame(bucket_rows).to_csv(output / "residual_volatility_buckets.csv", index=False)

    train_dispersion = panels["prior_dispersion"].loc[TRAIN[0] : TRAIN[1]].dropna()
    low, high = train_dispersion.quantile([1 / 3, 2 / 3])
    regimes = {
        "dispersion": pd.cut(
            panels["prior_dispersion"], [-np.inf, low, high, np.inf],
            labels=["low", "mid", "high"],
        ),
        "drawdown": pd.cut(
            panels["prior_drawdown"], [-np.inf, -0.15, -0.05, np.inf],
            labels=["stress_below_15pct", "drawdown_5_to_15pct", "normal_above_5pct"],
        ),
    }
    regime_rows = []
    for regime_type, regime in regimes.items():
        for segment, (start, end) in (("train", TRAIN), ("development", VALIDATION)):
            frame = daily.loc[start:end]
            labels_for_segment = regime.loc[start:end]
            for label in labels_for_segment.dropna().unique():
                selected = frame.loc[labels_for_segment.eq(label)]
                regime_rows.append(
                    {"regime_type": regime_type, "segment": segment, "regime": str(label),
                     "sessions": len(selected),
                     "mean_net_return_bps": float(selected.net_return.mean() * 10_000),
                     "mean_gross_return_bps": float(selected.gross_return.mean() * 10_000)}
                )
    pd.DataFrame(regime_rows).to_csv(output / "regime_attribution.csv", index=False)

    exposure = result.rebalances.copy()
    before_rows = []
    for session in exposure.index:
        usable = (
            panels["eligibility"].loc[session]
            & score.loc[session].notna()
            & panels["beta"].loc[session].notna()
        )
        names = score.columns[usable]
        raw = score.loc[session, names].rank(pct=True) - 0.5
        raw = raw * (0.95 / raw.abs().sum())
        sector_before = raw.groupby(panels["sectors"].loc[names]).sum().abs().max()
        beta_before = abs((raw * panels["beta"].loc[session, names]).sum())
        before_rows.append(
            {"session": session, "sector_before": sector_before,
             "sector_after": exposure.loc[session, "max_sector_exposure"],
             "beta_before": beta_before,
             "beta_after": abs(exposure.loc[session, "beta_exposure"])}
        )
    before_after = pd.DataFrame(before_rows).set_index("session")
    before_after["year"] = before_after.index.year
    before_after.groupby("year").mean().reset_index().to_csv(
        output / "neutralization_before_after.csv", index=False
    )
    exposure["year"] = exposure.index.year
    exposure.groupby("year").agg(
        max_abs_net=("net_exposure", lambda value: value.abs().max()),
        max_abs_beta=("beta_exposure", lambda value: value.abs().max()),
        max_abs_sector=("max_sector_exposure", "max"),
        mean_eligible_names=("eligible_names", "mean"),
    ).reset_index().to_csv(output / "exposure_checks_by_year.csv", index=False)
    (output / "diagnostic_candidate.txt").write_text(candidate + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("output/equities_2026-09-20_sip_r2"))
    parser.add_argument("--v2-output", type=Path, default=Path("reports/equity_v2/backtest"))
    parser.add_argument("--output", type=Path, default=Path("reports/equity_v9"))
    args = parser.parse_args()
    main(args.source, args.v2_output, args.output)

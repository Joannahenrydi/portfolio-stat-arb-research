"""Create post-selection diagnostics for the frozen v7 R01 baseline."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from backtest.residual_reversal import ResidualReversalConfig, fit_stress_artifact
from portfolio.optimizer import PortfolioCosts
from scripts.evaluate_equity_v2 import TRAIN, VALIDATION
from scripts.evaluate_equity_v7 import load_panels, run_candidate


def _segment(index: pd.DatetimeIndex) -> pd.Series:
    result = pd.Series("outside", index=index)
    result.loc[TRAIN[0] : TRAIN[1]] = "train"
    result.loc[VALIDATION[0] : VALIDATION[1]] = "development"
    return result


def holding_episodes(weights: pd.DataFrame, tolerance: float = 1e-10) -> pd.DataFrame:
    rows = []
    for symbol in weights:
        values = weights[symbol].to_numpy()
        signs = np.where(values > tolerance, 1, np.where(values < -tolerance, -1, 0))
        start = None
        active_sign = 0
        for offset, sign in enumerate(signs):
            if sign == active_sign:
                continue
            if active_sign != 0 and start is not None:
                rows.append(
                    {
                        "symbol": symbol,
                        "side": "long" if active_sign > 0 else "short",
                        "entry": weights.index[start],
                        "exit": weights.index[offset],
                        "sessions": offset - start,
                        "completed": True,
                    }
                )
            start = offset if sign != 0 else None
            active_sign = sign
        if active_sign != 0 and start is not None:
            rows.append(
                {
                    "symbol": symbol,
                    "side": "long" if active_sign > 0 else "short",
                    "entry": weights.index[start],
                    "exit": weights.index[-1],
                    "sessions": len(weights) - start,
                    "completed": False,
                }
            )
    return pd.DataFrame(rows)


def main(source: Path, v2_output: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    panels, _ = load_panels(source, v2_output, VALIDATION[1])
    artifact = fit_stress_artifact(panels["prior_market_volatility"], *TRAIN)
    result = run_candidate(
        panels["scores"]["core"], ResidualReversalConfig(), panels, artifact,
        PortfolioCosts(), VALIDATION[1]
    )
    if result.status != "COMPLETED":
        raise RuntimeError(f"R01 diagnostic replay failed: {result.reason}")
    daily = result.daily.copy()
    weights = result.weights.copy()
    sessions = daily.index
    returns = panels["returns"].loc[sessions]
    contributions = weights * returns
    long_pnl = contributions.where(weights.gt(0), 0).sum(axis=1)
    short_pnl = contributions.where(weights.lt(0), 0).sum(axis=1)
    daily["long_pnl"] = long_pnl
    daily["short_pnl"] = short_pnl
    daily["cost_drag"] = daily.transaction_cost + daily.borrow_cost
    daily["segment"] = _segment(sessions)
    daily.to_csv(output / "R01_daily.csv")
    weights.to_csv(output / "R01_weights.csv.gz", compression="gzip")

    annual_rows = []
    for year, frame in daily.groupby(daily.index.year):
        annual_rows.append(
            {
                "year": year,
                "net_return": float((1 + frame.net_return).prod() - 1),
                "gross_return": float((1 + frame.gross_return).prod() - 1),
                "transaction_cost": float(frame.transaction_cost.sum()),
                "borrow_cost": float(frame.borrow_cost.sum()),
                "cost_drag": float(frame.cost_drag.sum()),
                "turnover": float(frame.turnover.sum()),
                "long_pnl": float(frame.long_pnl.sum()),
                "short_pnl": float(frame.short_pnl.sum()),
                "average_gross": float(frame.gross.mean()),
            }
        )
    pd.DataFrame(annual_rows).to_csv(output / "annual_attribution.csv", index=False)

    episodes = holding_episodes(weights)
    episodes.to_csv(output / "holding_episodes.csv.gz", index=False, compression="gzip")
    completed = episodes.loc[episodes.completed].copy()
    completed["year"] = pd.to_datetime(completed.entry).dt.year
    holding = (
        completed.groupby(["year", "side"]).sessions.agg(["mean", "median", "count"])
        .reset_index()
    )
    holding.to_csv(output / "holding_period_by_year.csv", index=False)

    residual_vol = panels["residual_volatility"].loc[sessions]
    percentile = residual_vol.rank(axis=1, pct=True)
    bucket_rows = []
    for segment_name, (start, end) in (("train", TRAIN), ("development", VALIDATION)):
        years = len(daily.loc[start:end]) / 252
        for bucket in range(1, 6):
            lower, upper = (bucket - 1) / 5, bucket / 5
            mask = percentile.gt(lower) & percentile.le(upper)
            pnl = float(contributions.loc[start:end].where(mask.loc[start:end], 0).sum().sum())
            bucket_rows.append(
                {"segment": segment_name, "bucket": f"Q{bucket}", "pnl_total": pnl,
                 "pnl_annualized_additive": pnl / years}
            )
    pd.DataFrame(bucket_rows).to_csv(output / "residual_volatility_buckets.csv", index=False)

    dispersion = panels["prior_dispersion"].loc[sessions]
    train_dispersion = dispersion.loc[TRAIN[0] : TRAIN[1]].dropna()
    low, high = train_dispersion.quantile([1 / 3, 2 / 3])
    dispersion_regime = pd.cut(
        dispersion, [-np.inf, low, high, np.inf], labels=["low", "mid", "high"]
    )
    drawdown = panels["prior_drawdown"].loc[sessions]
    drawdown_regime = pd.cut(
        drawdown, [-np.inf, -0.15, -0.05, np.inf],
        labels=["stress_below_15pct", "drawdown_5_to_15pct", "normal_above_5pct"]
    )
    regime_rows = []
    for regime_type, labels in (
        ("dispersion", dispersion_regime), ("drawdown", drawdown_regime)
    ):
        for segment_name, (start, end) in (("train", TRAIN), ("development", VALIDATION)):
            frame = daily.loc[start:end]
            selected_labels = labels.loc[start:end]
            for label in selected_labels.dropna().unique():
                selected = frame.loc[selected_labels.eq(label)]
                regime_rows.append(
                    {
                        "regime_type": regime_type,
                        "segment": segment_name,
                        "regime": str(label),
                        "sessions": len(selected),
                        "mean_net_return_bps": float(selected.net_return.mean() * 10_000),
                        "mean_gross_return_bps": float(selected.gross_return.mean() * 10_000),
                        "mean_cost_bps": float(selected.cost_drag.mean() * 10_000),
                    }
                )
    pd.DataFrame(regime_rows).to_csv(output / "regime_attribution.csv", index=False)

    exposure = result.rebalances.copy()
    exposure["year"] = exposure.index.year
    exposure.groupby("year").agg(
        max_abs_net=("net_exposure", lambda value: value.abs().max()),
        max_abs_beta=("beta_exposure", lambda value: value.abs().max()),
        max_abs_sector=("max_sector_exposure", "max"),
        mean_eligible_names=("eligible_names", "mean"),
    ).reset_index().to_csv(output / "exposure_checks_by_year.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("output/equities_2026-09-20_sip_r2"))
    parser.add_argument("--v2-output", type=Path, default=Path("reports/equity_v2/backtest"))
    parser.add_argument("--output", type=Path, default=Path("reports/equity_v7"))
    args = parser.parse_args()
    main(args.source, args.v2_output, args.output)

"""Post-selection diagnostics for the frozen v8 sleeve allocator."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from scripts.evaluate_equity_v2 import TRAIN, VALIDATION
from scripts.evaluate_equity_v8 import load_sleeve_returns


def capped_maximum_mean(means: pd.Series, cap: float = 0.60) -> float:
    remaining = 1.0
    result = 0.0
    for value in means.sort_values(ascending=False):
        weight = min(cap, remaining)
        result += weight * value
        remaining -= weight
        if remaining <= 1e-12:
            break
    return result


def main(source: Path, v2_output: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    sleeves, benchmark, _ = load_sleeve_returns(source, v2_output, include_audit=False)
    first_fit = sleeves.iloc[:126]
    sleeve_means = first_fit.mean()
    maximum = capped_maximum_mean(sleeve_means)
    floors = pd.DataFrame(
        [
            {"floor": "SPY mean", "daily_mean": float(benchmark.iloc[:126].mean())},
            {"floor": "zero", "daily_mean": 0.0},
            {"floor": "equal-weight mean", "daily_mean": float(first_fit.mean(axis=1).mean())},
            {"floor": "maximum feasible", "daily_mean": maximum},
        ]
    )
    floors["daily_mean_bps"] = floors.daily_mean * 10_000
    floors.to_csv(output / "initial_return_floor_feasibility.csv", index=False)
    sleeve_means.rename("daily_mean").mul(10_000).rename("daily_mean_bps").to_csv(
        output / "initial_sleeve_means.csv"
    )

    annual_rows = []
    rolling_rows = []
    for experiment in ("S00", "S07"):
        daily = pd.read_csv(output / f"{experiment}_daily.csv", parse_dates=["session"])
        daily = daily.set_index("session")
        for year, frame in daily.loc[daily.index >= "2018-07-03"].groupby(
            daily.loc[daily.index >= "2018-07-03"].index.year
        ):
            annual_rows.append(
                {
                    "experiment": experiment,
                    "year": year,
                    "net_return": float((1 + frame.net_return).prod() - 1),
                    "allocation_cost": float(frame.allocation_cost.sum()),
                }
            )
        rolling = daily.net_return.rolling(63, min_periods=40).corr(benchmark.loc[daily.index])
        rolling_rows.extend(
            {"session": session, "experiment": experiment, "rolling_63d_correlation": value}
            for session, value in rolling.dropna().items()
        )
    pd.DataFrame(annual_rows).to_csv(output / "annual_performance.csv", index=False)
    pd.DataFrame(rolling_rows).to_csv(output / "rolling_market_correlation.csv", index=False)

    comparison_rows = []
    for experiment in ("S00", "S07"):
        daily = pd.read_csv(output / f"{experiment}_daily.csv", parse_dates=["session"])
        daily = daily.set_index("session")
        for segment_name, (start, end) in (("train", TRAIN), ("development", VALIDATION)):
            section = daily.loc[max(start, pd.Timestamp("2018-07-03")) : end].copy()
            nav = (1 + section.net_return).cumprod()
            comparison_rows.extend(
                {
                    "session": session,
                    "experiment": experiment,
                    "segment": segment_name,
                    "nav": value,
                }
                for session, value in nav.items()
            )
    pd.DataFrame(comparison_rows).to_csv(output / "cumulative_performance.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("output/equities_2026-09-20_sip_r2"))
    parser.add_argument("--v2-output", type=Path, default=Path("reports/equity_v2/backtest"))
    parser.add_argument("--output", type=Path, default=Path("reports/equity_v8"))
    args = parser.parse_args()
    main(args.source, args.v2_output, args.output)

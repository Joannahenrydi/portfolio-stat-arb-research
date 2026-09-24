"""Render v8 allocation and paper-objective diagnostics."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def save(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main(folder: Path) -> None:
    candidates = pd.read_csv(folder / "candidate_selection.csv")
    completed = candidates[candidates.status.isin(["REFERENCE_ONLY", "COMPLETED"])]
    x = np.arange(len(completed))
    width = 0.36
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, completed.train_sharpe, width, label="Train", color="#2563eb")
    ax.bar(
        x + width / 2, completed.development_sharpe, width,
        label="Development", color="#ea580c",
    )
    ax.axhline(0.70, color="#2563eb", linestyle="--", linewidth=1)
    ax.axhline(0.50, color="#ea580c", linestyle=":", linewidth=1.5)
    ax.set_xticks(x, completed.experiment)
    ax.set_ylabel("Net Sharpe")
    ax.set_title("Only S00 and S07 remained feasible")
    ax.legend()
    save(fig, folder / "01_candidate_sharpe.png")

    singles = pd.read_csv(folder / "single_sleeve_comparison.csv")
    pivot = singles.pivot(index="sleeve", columns="segment", values="cagr") * 100
    fig, ax = plt.subplots(figsize=(9, 5))
    pivot[["train", "development"]].plot.bar(ax=ax, color=["#2563eb", "#ea580c"])
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Net CAGR (%)")
    ax.set_title("Every input sleeve lost money after modeled costs")
    ax.legend(title="")
    save(fig, folder / "02_single_sleeve_cagr.png")

    floors = pd.read_csv(folder / "initial_return_floor_feasibility.csv")
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#dc2626", "#dc2626", "#16a34a", "#2563eb"]
    ax.bar(floors.floor, floors.daily_mean_bps, color=colors)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("First-fit daily mean (bps)")
    ax.set_title("Paper and zero return floors were infeasible at the first refit")
    ax.tick_params(axis="x", rotation=15)
    save(fig, folder / "03_return_floor_feasibility.png")

    refits = pd.read_csv(folder / "S07_refits.csv", parse_dates=["session"])
    weight_columns = [column for column in refits if column.startswith("weight_")]
    labels = [column.removeprefix("weight_") for column in weight_columns]
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.stackplot(
        refits.session, *[refits[column] for column in weight_columns], labels=labels, alpha=0.9
    )
    ax.set_ylim(0, 1)
    ax.set_ylabel("Allocation weight")
    ax.set_title("S07 walk-forward strategy-sleeve weights")
    ax.legend(loc="upper left", ncol=3, fontsize=8)
    save(fig, folder / "04_S07_weight_path.png")

    rolling = pd.read_csv(folder / "rolling_market_correlation.csv", parse_dates=["session"])
    fig, ax = plt.subplots(figsize=(11, 5))
    for experiment, frame in rolling.groupby("experiment"):
        ax.plot(frame.session, frame.rolling_63d_correlation, label=experiment)
    ax.axhline(0.20, color="black", linestyle="--", linewidth=1)
    ax.axhline(-0.20, color="black", linestyle="--", linewidth=1)
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_ylabel("Rolling 63-session correlation with SPY")
    ax.set_title("Full-period neutrality can hide unstable short-window correlation")
    ax.legend()
    save(fig, folder / "05_rolling_market_correlation.png")

    cumulative = pd.read_csv(folder / "cumulative_performance.csv", parse_dates=["session"])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, segment in zip(axes, ("train", "development")):
        for experiment, frame in cumulative[cumulative.segment.eq(segment)].groupby("experiment"):
            ax.plot(frame.session, frame.nav, label=experiment)
        ax.set_title(segment.title())
        ax.set_ylabel("Growth of $1")
        ax.legend()
    fig.suptitle("Equal weight versus paper-inspired S07")
    save(fig, folder / "06_cumulative_performance.png")

    fig, ax = plt.subplots(figsize=(10, 5))
    for experiment in ("S00", "S07"):
        frame = pd.read_csv(folder / f"{experiment}_refits.csv", parse_dates=["session"])
        ax.plot(frame.session, frame.absolute_correlation, marker="o", label=experiment)
    ax.set_ylabel("In-sample absolute correlation")
    ax.set_title("Objective value at each walk-forward refit")
    ax.legend()
    save(fig, folder / "07_refit_objective_correlation.png")

    annual = pd.read_csv(folder / "annual_performance.csv")
    pivot = annual.pivot(index="year", columns="experiment", values="net_return") * 100
    fig, ax = plt.subplots(figsize=(9, 5))
    pivot.plot.bar(ax=ax, color=["#64748b", "#7c3aed"])
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Net return (%)")
    ax.set_title("Annual net performance")
    ax.legend(title="")
    save(fig, folder / "08_annual_performance.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=Path, default=Path("reports/equity_v8"))
    args = parser.parse_args()
    main(args.folder)

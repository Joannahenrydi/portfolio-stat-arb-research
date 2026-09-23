"""Render separate, reviewable v7 diagnostic figures."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


COLORS = {"train": "#2563eb", "development": "#ea580c"}


def _save(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main(folder: Path) -> None:
    candidates = pd.read_csv(folder / "candidate_selection.csv")
    candidates = candidates[candidates.experiment.ne("R00")].sort_values("experiment")
    valid = candidates.status.eq("COMPLETED")

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(candidates))
    width = 0.36
    ax.bar(x - width / 2, candidates.train_sharpe, width, label="Train", color=COLORS["train"])
    ax.bar(
        x + width / 2, candidates.development_sharpe, width,
        label="Development holdout", color=COLORS["development"]
    )
    ax.axhline(0.70, color=COLORS["train"], linestyle="--", linewidth=1, label="Train gate 0.70")
    ax.axhline(0.50, color=COLORS["development"], linestyle=":", linewidth=1.5,
               label="Development gate 0.50")
    ax.set_xticks(x, candidates.experiment)
    ax.set_ylabel("Net Sharpe")
    ax.set_title("v7 candidate Sharpe versus frozen profitability gates")
    ax.legend(ncol=2, fontsize=8)
    _save(fig, folder / "01_candidate_net_sharpe.png")

    fig, ax = plt.subplots(figsize=(9, 5))
    completed = candidates.loc[valid]
    x = np.arange(len(completed))
    ax.bar(x - width / 2, completed.development_gross_sharpe, width,
           label="Gross", color="#16a34a")
    ax.bar(x + width / 2, completed.development_sharpe, width,
           label="Net", color="#dc2626")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x, completed.experiment)
    ax.set_ylabel("Development Sharpe")
    ax.set_title("Gross signal quality does not survive modeled costs")
    ax.legend()
    _save(fig, folder / "02_gross_vs_net_sharpe.png")

    decay = pd.read_csv(folder / "signal_decay.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharex=True)
    for segment, frame in decay.groupby("segment"):
        color = COLORS[segment]
        axes[0].plot(frame.horizon, frame.mean_ic, marker="o", label=segment, color=color)
        axes[1].plot(frame.horizon, frame.mean_rank_ic, marker="o", label=segment, color=color)
    for ax, title in zip(axes, ("Pearson IC", "Rank IC")):
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title(title)
        ax.set_xlabel("Forward sessions")
        ax.legend()
    axes[0].set_ylabel("Mean daily IC")
    fig.suptitle("Core alpha decay by evidence interval")
    _save(fig, folder / "03_alpha_ic_decay.png")

    sides = pd.read_csv(folder / "long_short_ic.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, segment in zip(axes, ("train", "development")):
        frame = sides[sides.segment.eq(segment)]
        ax.plot(frame.horizon, frame.long_rank_ic, marker="o", label="Long", color="#16a34a")
        ax.plot(frame.horizon, frame.short_rank_ic, marker="o", label="Short", color="#dc2626")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title(segment.title())
        ax.set_xlabel("Forward sessions")
        ax.legend()
    axes[0].set_ylabel("Mean rank IC")
    fig.suptitle("Long and short predictive content")
    _save(fig, folder / "04_long_short_ic.png")

    annual = pd.read_csv(folder / "annual_attribution.csv")
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(annual))
    ax.bar(x - width / 2, annual.gross_return * 100, width, label="Gross return", color="#16a34a")
    ax.bar(x + width / 2, annual.net_return * 100, width, label="Net return", color="#dc2626")
    ax.set_xticks(x, annual.year)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Annual return (%)")
    ax.set_title("R01 annual gross and net performance")
    ax.legend()
    _save(fig, folder / "05_annual_gross_net.png")

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width / 2, annual.long_pnl * 100, width, label="Long leg", color="#16a34a")
    ax.bar(x + width / 2, annual.short_pnl * 100, width, label="Short leg", color="#dc2626")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x, annual.year)
    ax.set_ylabel("Additive PnL contribution (% NAV)")
    ax.set_title("R01 long and short leg attribution")
    ax.legend()
    _save(fig, folder / "06_long_short_pnl.png")

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x, annual.transaction_cost * 100, label="Transaction cost", color="#f59e0b")
    ax.bar(x, annual.borrow_cost * 100, bottom=annual.transaction_cost * 100,
           label="Borrow", color="#7c3aed")
    ax.set_xticks(x, annual.year)
    ax.set_ylabel("Annual drag (% NAV)")
    ax.set_title("R01 modeled cost drag")
    ax.legend()
    _save(fig, folder / "07_cost_drag.png")

    holding = pd.read_csv(folder / "holding_period_by_year.csv")
    pivot = holding.pivot(index="year", columns="side", values="mean")
    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.plot(pivot.index, pivot.long, marker="o", label="Long holding", color="#16a34a")
    ax1.plot(pivot.index, pivot.short, marker="o", label="Short holding", color="#dc2626")
    ax1.set_ylabel("Mean completed holding (sessions)")
    ax2 = ax1.twinx()
    ax2.bar(annual.year, annual.turnover, alpha=0.20, label="Turnover", color="#2563eb")
    ax2.set_ylabel("Annual turnover (x NAV)")
    ax1.set_title("R01 holding duration and turnover")
    lines, labels = ax1.get_legend_handles_labels()
    bars, bar_labels = ax2.get_legend_handles_labels()
    ax1.legend(lines + bars, labels + bar_labels, loc="best")
    _save(fig, folder / "08_turnover_holding.png")

    buckets = pd.read_csv(folder / "residual_volatility_buckets.csv")
    pivot = buckets.pivot(index="bucket", columns="segment", values="pnl_annualized_additive")
    fig, ax = plt.subplots(figsize=(8, 5))
    pivot[["train", "development"]].mul(100).plot.bar(
        ax=ax, color=[COLORS["train"], COLORS["development"]]
    )
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Annualized additive gross PnL (% NAV)")
    ax.set_title("Residual-volatility bucket attribution")
    ax.legend(title="")
    _save(fig, folder / "09_residual_volatility_buckets.png")

    regimes = pd.read_csv(folder / "regime_attribution.csv")
    for number, regime_type, title in (
        (10, "dispersion", "Cross-sectional dispersion regime"),
        (11, "drawdown", "Market drawdown regime"),
    ):
        frame = regimes[regimes.regime_type.eq(regime_type)]
        pivot = frame.pivot(index="regime", columns="segment", values="mean_net_return_bps")
        fig, ax = plt.subplots(figsize=(9, 5))
        pivot.reindex(columns=["train", "development"]).plot.bar(
            ax=ax, color=[COLORS["train"], COLORS["development"]]
        )
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_ylabel("Mean daily net return (bps)")
        ax.set_title(title)
        ax.legend(title="")
        _save(fig, folder / f"{number:02d}_{regime_type}_regime.png")

    exposure = pd.read_csv(folder / "exposure_checks_by_year.csv")
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.semilogy(exposure.year, exposure.max_abs_net, marker="o", label="Net")
    ax.semilogy(exposure.year, exposure.max_abs_beta, marker="o", label="Beta")
    ax.semilogy(exposure.year, exposure.max_abs_sector, marker="o", label="Sector")
    ax.axhline(1e-8, color="black", linestyle="--", label="Tolerance 1e-8")
    ax.set_ylabel("Maximum absolute exposure")
    ax.set_title("R01 post-trade neutrality checks")
    ax.legend()
    _save(fig, folder / "12_neutrality_checks.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=Path, default=Path("reports/equity_v7"))
    args = parser.parse_args()
    main(args.folder)

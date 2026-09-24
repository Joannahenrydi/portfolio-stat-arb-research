"""Render separate v9 figures for the frozen rejected experiment."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, ORANGE, GREEN, RED = "#2563eb", "#ea580c", "#16a34a", "#dc2626"


def save(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def grouped(frame: pd.DataFrame, title: str, ylabel: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    frame.plot.bar(ax=ax, color=[BLUE, ORANGE, GREEN, RED][: len(frame.columns)])
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=0)
    save(fig, path)


def main(folder: Path) -> None:
    candidate = (folder / "diagnostic_candidate.txt").read_text().strip()
    ic = pd.read_csv(folder / "diagnostic_ic_by_year.csv").set_index("year")
    grouped(
        ic.rename(columns={"ic": "IC", "rank_ic": "Rank IC"}),
        f"{candidate} alpha IC and rank IC by year", "Mean daily 3-session IC",
        folder / "01_alpha_ic_rank_ic_by_year.png",
    )
    annual = pd.read_csv(folder / "annual_attribution.csv").set_index("year")
    grouped(
        annual[["gross_sharpe", "net_sharpe"]].rename(
            columns={"gross_sharpe": "Gross", "net_sharpe": "Net"}
        ),
        f"{candidate} gross Sharpe versus net Sharpe", "Annualized Sharpe",
        folder / "02_gross_vs_net_sharpe.png",
    )
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(annual.index.astype(str), annual.turnover, color=BLUE)
    ax.set_title(f"{candidate} turnover by year")
    ax.set_ylabel("Sum absolute traded weight (x NAV)")
    save(fig, folder / "03_turnover_by_year.png")

    holding = pd.read_csv(folder / "holding_period_by_year.csv")
    pivot = holding.pivot(index="year", columns="side", values="mean")
    grouped(
        pivot.reindex(columns=["long", "short"]), f"{candidate} average holding period",
        "Completed holding episode (sessions)", folder / "04_average_holding_period.png",
    )
    grouped(
        annual[["long_pnl", "short_pnl"]].mul(100).rename(
            columns={"long_pnl": "Long leg", "short_pnl": "Short leg"}
        ),
        f"{candidate} long and short leg PnL", "Additive gross PnL (% NAV)",
        folder / "05_long_short_leg_pnl.png",
    )
    neutral = pd.read_csv(folder / "neutralization_before_after.csv").set_index("year")
    grouped(
        neutral[["sector_before", "sector_after"]].rename(
            columns={"sector_before": "Before", "sector_after": "After"}
        ),
        "Sector exposure before and after neutralization", "Mean maximum sector exposure",
        folder / "06_sector_neutral_before_after.png",
    )
    grouped(
        neutral[["beta_before", "beta_after"]].rename(
            columns={"beta_before": "Before", "beta_after": "After"}
        ),
        "Beta exposure before and after neutralization", "Mean absolute beta exposure",
        folder / "07_beta_neutral_before_after.png",
    )
    fig, ax = plt.subplots(figsize=(9, 5))
    annual[["transaction_cost", "borrow_cost"]].mul(10_000).rename(
        columns={"transaction_cost": "Trading", "borrow_cost": "Borrow"}
    ).plot.bar(stacked=True, ax=ax, color=[ORANGE, RED])
    ax.set_title(f"{candidate} modeled cost drag")
    ax.set_ylabel("Cumulative bps of NAV")
    ax.tick_params(axis="x", rotation=0)
    save(fig, folder / "08_cost_drag.png")

    buckets = pd.read_csv(folder / "residual_volatility_buckets.csv")
    grouped(
        buckets.pivot(index="bucket", columns="segment", values="pnl_annualized_additive")
        .reindex(columns=["train", "development"]).mul(100),
        "Residual-volatility bucket attribution", "Annualized additive gross PnL (% NAV)",
        folder / "09_residual_volatility_buckets.png",
    )
    regimes = pd.read_csv(folder / "regime_attribution.csv")
    for number, kind, title in (
        (10, "dispersion", "Cross-sectional dispersion regime"),
        (11, "drawdown", "Market drawdown regime"),
    ):
        frame = regimes.loc[regimes.regime_type.eq(kind)]
        grouped(
            frame.pivot(index="regime", columns="segment", values="mean_net_return_bps")
            .reindex(columns=["train", "development"]),
            title, "Mean daily net return (bps)", folder / f"{number:02d}_{kind}_regime.png",
        )

    qr = pd.read_csv(folder / "adaptive_qr_distributions.csv")
    qr_dev = qr.loc[qr.segment.eq("development")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, field in zip(axes, ("q_prior", "r_prior")):
        frame = qr_dev.loc[qr_dev.field.eq(field)]
        x = np.arange(len(frame))
        ax.bar(x - 0.18, frame.p50, 0.36, label="Median", color=BLUE)
        ax.bar(x + 0.18, frame.p99, 0.36, label="P99", color=ORANGE)
        ax.set_xticks(x, frame.model)
        ax.set_yscale("log")
        ax.set_title(field)
        ax.legend()
    fig.suptitle("Adaptive Kalman prior variance distributions")
    save(fig, folder / "12_adaptive_qr_distributions.png")

    confidence = pd.read_csv(folder / "confidence_filter_by_year.csv")
    mean_pass = confidence.groupby("experiment").pass_rate.mean().mul(100)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(mean_pass.index, mean_pass, color=BLUE)
    ax.set_title("Cost-confidence filter pass rate")
    ax.set_ylabel("Mean eligible observations (%)")
    save(fig, folder / "13_confidence_filter_pass_rate.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=Path, default=Path("reports/equity_v9"))
    args = parser.parse_args()
    main(args.folder)

"""Plot frozen v5 development results without reading the reused audit."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE = "#2563EB"
ORANGE = "#F59E0B"
GREEN = "#059669"
RED = "#DC2626"


def finish(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main(source: Path) -> None:
    plt.rcParams.update(
        {"axes.spines.top": False, "axes.spines.right": False, "axes.titleweight": "bold"}
    )
    candidates = pd.read_csv(source / "candidate_selection.csv").set_index("experiment")
    selectable = candidates.drop(index="E00").sort_index()
    positions = np.arange(len(selectable))
    width = 0.38
    fig, ax = plt.subplots(figsize=(11, 5.8))
    ax.bar(positions - width / 2, selectable.train_sharpe, width, label="Train", color=BLUE)
    ax.bar(
        positions + width / 2,
        selectable.validation_sharpe,
        width,
        label="Validation",
        color=ORANGE,
    )
    ax.axhline(0, color="#0F172A", linewidth=0.8)
    ax.set(
        title="v5 Candidate Net Sharpe: Train vs Validation",
        xlabel="Frozen experiment",
        ylabel="Annualized Sharpe",
        xticks=positions,
        xticklabels=selectable.index,
    )
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.2)
    finish(fig, source / "01_candidate_train_validation_sharpe.png")

    fig, ax = plt.subplots(figsize=(11, 5.8))
    colors = [GREEN if value > 0 else RED for value in selectable.validation_cagr]
    ax.bar(selectable.index, selectable.validation_cagr * 100, color=colors)
    ax.axhline(0, color="#0F172A", linewidth=0.8)
    ax.set(
        title="v5 Validation CAGR",
        xlabel="Frozen experiment",
        ylabel="Net CAGR (%)",
    )
    ax.grid(axis="y", alpha=0.2)
    finish(fig, source / "02_validation_cagr.png")

    decay = pd.read_csv(source / "signal_decay.csv")
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for segment, color in (("train", BLUE), ("validation", ORANGE)):
        section = decay.loc[decay.segment.eq(segment)]
        ax.plot(
            section.horizon,
            section.mean_rank_ic,
            marker="o",
            linewidth=2,
            label=segment.title(),
            color=color,
        )
    ax.axhline(0, color="#0F172A", linewidth=0.8)
    ax.set(
        title="Baseline Signal Decay",
        xlabel="Forward horizon (sessions)",
        ylabel="Mean cross-sectional rank IC",
        xticks=[1, 5, 10, 21, 42],
    )
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)
    finish(fig, source / "03_signal_decay.png")

    leg = pd.read_csv(source / "long_short_ic.csv")
    validation = leg.loc[leg.segment.eq("validation")]
    positions = np.arange(len(validation))
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.bar(positions - width / 2, validation.long_rank_ic, width, label="Long", color=BLUE)
    ax.bar(positions + width / 2, validation.short_rank_ic, width, label="Short", color=RED)
    ax.axhline(0, color="#0F172A", linewidth=0.8)
    ax.set(
        title="Validation Long vs Short Rank IC",
        xlabel="Forward horizon (sessions)",
        ylabel="Mean rank IC",
        xticks=positions,
        xticklabels=validation.horizon,
    )
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.2)
    finish(fig, source / "04_validation_long_short_ic.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("reports/equity_v5"))
    main(parser.parse_args().source)

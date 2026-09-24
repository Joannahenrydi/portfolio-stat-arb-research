"""Plot the frozen v10 universe-expansion result."""

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

FOLDER = Path("reports/equity_v10")


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FOLDER / name, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    frame = pd.read_csv(FOLDER / "candidate_selection.csv").set_index("experiment")
    selectable = frame.loc[frame.index != "X00"]
    x = np.arange(len(selectable))
    width = 0.36
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2, selectable.train_sharpe, width, label="Train", color="#2563eb")
    ax.bar(x + width / 2, selectable.development_sharpe, width, label="Development", color="#ea580c")
    ax.axhline(0.70, color="#2563eb", linestyle="--", linewidth=1)
    ax.axhline(0.50, color="#ea580c", linestyle=":", linewidth=1)
    ax.set_xticks(x, selectable.index)
    ax.set_ylabel("Net Sharpe")
    ax.set_title("v10 net Sharpe versus frozen gates")
    ax.legend()
    save(fig, "01_net_sharpe_gates.png")

    completed = selectable.loc[selectable.status.eq("COMPLETED")]
    x = np.arange(len(completed))
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width / 2, completed.development_gross_sharpe, width, label="Gross", color="#16a34a")
    ax.bar(x + width / 2, completed.development_sharpe, width, label="Net", color="#dc2626")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x, completed.index)
    ax.set_ylabel("Development Sharpe")
    ax.set_title("Expanded-universe gross alpha versus net result")
    ax.legend()
    save(fig, "02_development_gross_net_sharpe.png")

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x, completed.development_annual_turnover, color="#2563eb")
    ax.set_xticks(x, completed.index)
    ax.set_ylabel("Annual turnover (x NAV)")
    ax.set_title("Slower rebalancing reduces turnover")
    save(fig, "03_turnover.png")

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x, completed.development_average_eligible_names, color="#7c3aed")
    ax.axhline(100, color="black", linestyle="--", label="Acceptance minimum")
    ax.set_xticks(x, completed.index)
    ax.set_ylabel("Average eligible names")
    ax.set_title("Tail concentration and executable breadth")
    ax.legend()
    save(fig, "04_eligible_breadth.png")


if __name__ == "__main__":
    main()

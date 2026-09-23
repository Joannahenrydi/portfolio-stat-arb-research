"""Create separate diagnostic figures for the locked market-neutral portfolio."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from backtest.engine import (
    FastPortfolioConfig,
    candidate_scores,
    residual_returns,
    run_fast_backtest,
)
from portfolio.optimizer import PortfolioCosts
from scripts.evaluate_equity_v2 import TRAIN, read_panel

AUDIT_END = pd.Timestamp("2026-09-18")
BLUE = "#2563EB"
ORANGE = "#F59E0B"
GREEN = "#059669"
RED = "#DC2626"
SLATE = "#64748B"


def _finish(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _grouped_bars(frame: pd.DataFrame, title: str, ylabel: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.2))
    frame.plot(kind="bar", ax=ax, color=[BLUE, ORANGE, GREEN, RED][: len(frame.columns)])
    ax.axhline(0, color="#0F172A", linewidth=0.8)
    ax.set(title=title, xlabel="Year", ylabel=ylabel)
    ax.tick_params(axis="x", rotation=0)
    ax.grid(axis="y", alpha=0.2)
    _finish(fig, path)


def _sharpe(series: pd.Series) -> float:
    clean = series.dropna()
    std = clean.std(ddof=1)
    return float(clean.mean() / std * np.sqrt(252)) if len(clean) > 1 and std > 0 else np.nan


def _annual_table(daily: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for year, section in daily.groupby(daily.index.year):
        rows.append(
            {
                "year": int(year),
                "gross_sharpe": _sharpe(section.gross_return),
                "net_sharpe": _sharpe(section.net_return),
                "turnover": float(section.turnover.sum()),
                "transaction_cost_bps": float(section.transaction_cost.sum() * 10_000),
                "borrow_cost_bps": float(section.borrow_cost.sum() * 10_000),
            }
        )
    return pd.DataFrame(rows).set_index("year")


def _holding_periods(weights: pd.DataFrame) -> pd.DataFrame:
    records = []
    signs = np.sign(weights.where(weights.abs() > 1e-10, 0).to_numpy())
    for column, symbol in enumerate(weights.columns):
        active_sign = 0
        start = None
        for offset, sign in enumerate(signs[:, column]):
            if sign == active_sign:
                continue
            if active_sign != 0 and start is not None:
                records.append(
                    {
                        "symbol": symbol,
                        "start_year": int(weights.index[start].year),
                        "sessions": offset - start,
                    }
                )
            active_sign = sign
            start = offset if sign != 0 else None
    return pd.DataFrame(records)


def _regime_stats(returns: pd.Series, labels: pd.Series, order: list[str]) -> pd.DataFrame:
    rows = []
    for label in order:
        values = returns[labels.eq(label)].dropna()
        rows.append(
            {
                "regime": label,
                "sessions": len(values),
                "net_sharpe": _sharpe(values),
                "annualized_return": float(values.mean() * 252) if len(values) else np.nan,
            }
        )
    return pd.DataFrame(rows).set_index("regime")


def _plot_regime(table: pd.DataFrame, title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5.2))
    bars = ax.bar(table.index, table.net_sharpe, color=[BLUE, ORANGE, RED][: len(table)])
    ax.axhline(0, color="#0F172A", linewidth=0.8)
    ax.set(title=title, xlabel="Regime", ylabel="Net Sharpe")
    ax.grid(axis="y", alpha=0.2)
    for bar, (_, row) in zip(bars, table.iterrows()):
        ax.annotate(
            f"n={int(row.sessions)}\nann. {row.annualized_return:.1%}",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 5 if bar.get_height() >= 0 else -28),
            textcoords="offset points",
            ha="center",
            fontsize=8,
        )
    _finish(fig, path)


def main(source: Path, quality: Path, selection_path: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
            "font.size": 10,
        }
    )
    universe = pd.read_csv(quality / "prospective_400_stocks.csv").sort_values("security_id")
    symbols = universe.symbol.tolist()
    sectors = universe.set_index("symbol").sector.reindex(symbols)
    sessions = pd.DatetimeIndex(pd.to_datetime(pd.read_csv(source / "market_calendar.csv").date))
    sessions = sessions[(sessions >= "2017-01-03") & (sessions <= AUDIT_END)]
    adjusted = read_panel(source / "bars_all.csv", symbols + ["SPY"], sessions, "c")
    study = sessions[(sessions >= TRAIN[0])]
    complete = adjusted.loc[study, symbols].notna().all()
    all_returns = adjusted[symbols].pct_change(fill_method=None)
    discontinuity = all_returns.loc[study].abs().gt(0.50).any()
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
    mask = returns.index >= TRAIN[0]
    config = FastPortfolioConfig()
    daily, weights = run_fast_backtest(
        score.loc[mask],
        returns.loc[mask],
        beta.loc[mask],
        sectors,
        raw_close.loc[mask],
        raw_volume.loc[mask],
        config=config,
        costs=PortfolioCosts(),
    )
    daily = daily.set_index("session") if "session" in daily.columns else daily

    # 1. Daily cross-sectional linear IC and Spearman rank IC, signal t versus return t+1.
    future = returns.shift(-1).reindex(score.index)
    ic = pd.DataFrame(
        {
            "Alpha IC": score.corrwith(future, axis=1, method="pearson"),
            "Rank IC": score.corrwith(future, axis=1, method="spearman"),
        }
    ).loc[TRAIN[0] : AUDIT_END]
    ic_year = ic.groupby(ic.index.year).mean()
    _grouped_bars(
        ic_year,
        "Alpha IC and Rank IC by Year",
        "Mean daily cross-sectional correlation",
        output / "01_alpha_ic_rank_ic_by_year.png",
    )

    annual = _annual_table(daily)
    _grouped_bars(
        annual[["gross_sharpe", "net_sharpe"]].rename(
            columns={"gross_sharpe": "Gross", "net_sharpe": "Net"}
        ),
        "Gross Sharpe vs Net Sharpe",
        "Annualized Sharpe",
        output / "02_gross_vs_net_sharpe.png",
    )

    fig, ax = plt.subplots(figsize=(10, 5.2))
    ax.bar(annual.index.astype(str), annual.turnover, color=BLUE)
    ax.set(title="Turnover by Year", xlabel="Year", ylabel="Sum of absolute traded weight")
    ax.grid(axis="y", alpha=0.2)
    _finish(fig, output / "03_turnover_by_year.png")

    periods = _holding_periods(weights)
    holding = periods.groupby("start_year").sessions.mean().to_frame("Mean holding period")
    fig, ax = plt.subplots(figsize=(10, 5.2))
    ax.bar(holding.index.astype(str), holding.iloc[:, 0], color=GREEN)
    ax.axhline(periods.sessions.mean(), color=SLATE, linestyle="--", label="Full-sample mean")
    ax.set(title="Average Completed Holding Period", xlabel="Position start year", ylabel="Sessions")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.2)
    _finish(fig, output / "04_average_holding_period.png")

    realized = returns.reindex(weights.index)[weights.columns]
    long_pnl = (weights.clip(lower=0) * realized).sum(axis=1)
    short_pnl = (weights.clip(upper=0) * realized).sum(axis=1)
    leg = pd.DataFrame({"Long leg": long_pnl, "Short leg": short_pnl})
    leg_year = leg.groupby(leg.index.year).sum() * 100
    _grouped_bars(
        leg_year,
        "Long Leg / Short Leg Gross PnL",
        "Arithmetic contribution (%)",
        output / "05_long_short_leg_pnl.png",
    )

    # 6–7. Compare raw rank portfolio with the next-session neutralized target.
    adv = (raw_close * raw_volume).rolling(config.adv_window, min_periods=config.adv_window).median()
    neutral_rows = []
    decision_index = score.loc[mask].index
    for offset in range(0, len(decision_index) - 1, config.rebalance_every):
        decision, effective = decision_index[offset], decision_index[offset + 1]
        usable = score.loc[decision].notna() & beta.loc[decision].notna() & adv.loc[decision].gt(0)
        names = score.columns[usable]
        if len(names) < 50:
            continue
        raw = score.loc[decision, names].rank(method="average", pct=True) - 0.5
        raw /= raw.abs().sum()
        after = weights.loc[effective]
        raw_sector = raw.groupby(sectors[names]).sum()
        after_sector = after.groupby(sectors).sum()
        neutral_rows.append(
            {
                "session": decision,
                "sector_before": float(raw_sector.abs().sum()),
                "sector_after": float(after_sector.abs().sum()),
                "beta_before": float(abs((raw * beta.loc[decision, names]).sum())),
                "beta_after": float(abs((after * beta.loc[decision]).sum())),
            }
        )
    neutral = pd.DataFrame(neutral_rows).set_index("session")
    sector_year = neutral.groupby(neutral.index.year)[["sector_before", "sector_after"]].mean()
    _grouped_bars(
        sector_year.rename(columns={"sector_before": "Before", "sector_after": "After"}),
        "Sector Exposure Before vs After Neutralization",
        "Mean sum of absolute sector weights",
        output / "06_sector_neutral_before_after.png",
    )
    beta_year = neutral.groupby(neutral.index.year)[["beta_before", "beta_after"]].mean()
    _grouped_bars(
        beta_year.rename(columns={"beta_before": "Before", "beta_after": "After"}),
        "Market Beta Exposure Before vs After Neutralization",
        "Mean absolute beta exposure",
        output / "07_beta_neutral_before_after.png",
    )

    cost = annual[["transaction_cost_bps", "borrow_cost_bps"]].rename(
        columns={"transaction_cost_bps": "Trading", "borrow_cost_bps": "Borrow"}
    )
    fig, ax = plt.subplots(figsize=(10, 5.2))
    cost.plot(kind="bar", stacked=True, ax=ax, color=[ORANGE, RED])
    ax.set(title="Cost Drag by Year", xlabel="Year", ylabel="Cumulative cost (bps of NAV)")
    ax.tick_params(axis="x", rotation=0)
    ax.grid(axis="y", alpha=0.2)
    _finish(fig, output / "08_cost_drag.png")

    residual_vol = residual.rolling(60, min_periods=60).std().shift(1).reindex(weights.index)
    contributions = weights * realized
    bucket_pnl = pd.DataFrame(index=weights.index, columns=[f"Q{i}" for i in range(1, 6)], dtype=float)
    for session in weights.index:
        ranks = residual_vol.loc[session].rank(pct=True)
        buckets = np.ceil(ranks * 5).clip(1, 5)
        for number in range(1, 6):
            bucket_pnl.at[session, f"Q{number}"] = contributions.loc[session, buckets.eq(number)].sum()
    bucket_summary = (bucket_pnl.mean() * 252 * 100).to_frame("Annualized contribution")
    fig, ax = plt.subplots(figsize=(8, 5.2))
    ax.bar(bucket_summary.index, bucket_summary.iloc[:, 0], color=["#DBEAFE", "#93C5FD", BLUE, "#1D4ED8", "#1E3A8A"])
    ax.axhline(0, color="#0F172A", linewidth=0.8)
    ax.set(
        title="PnL by Prior Residual-Volatility Bucket",
        xlabel="Q1 lowest volatility → Q5 highest volatility",
        ylabel="Annualized gross PnL contribution (%)",
    )
    ax.grid(axis="y", alpha=0.2)
    _finish(fig, output / "09_residual_volatility_bucket.png")

    prior_dispersion = residual.std(axis=1).shift(1).reindex(daily.index)
    dispersion_training = prior_dispersion.loc[TRAIN[0] : TRAIN[1]].dropna()
    low_cut, high_cut = dispersion_training.quantile([1 / 3, 2 / 3])
    dispersion_regime = pd.Series("Mid", index=daily.index)
    dispersion_regime.loc[prior_dispersion <= low_cut] = "Low"
    dispersion_regime.loc[prior_dispersion > high_cut] = "High"
    dispersion_regime.loc[prior_dispersion.isna()] = np.nan
    dispersion_stats = _regime_stats(daily.net_return, dispersion_regime, ["Low", "Mid", "High"])
    _plot_regime(
        dispersion_stats,
        "Net Performance by Prior Cross-Sectional Dispersion Regime",
        output / "10_cross_sectional_dispersion_regime.png",
    )

    market_drawdown = adjusted.SPY.div(adjusted.SPY.cummax()).sub(1).shift(1).reindex(daily.index)
    drawdown_regime = pd.Series("Normal (> -5%)", index=daily.index)
    drawdown_regime.loc[market_drawdown.le(-0.05) & market_drawdown.gt(-0.15)] = "Drawdown (-5% to -15%)"
    drawdown_regime.loc[market_drawdown.le(-0.15)] = "Stress (≤ -15%)"
    drawdown_regime.loc[market_drawdown.isna()] = np.nan
    drawdown_order = ["Normal (> -5%)", "Drawdown (-5% to -15%)", "Stress (≤ -15%)"]
    drawdown_stats = _regime_stats(daily.net_return, drawdown_regime, drawdown_order)
    _plot_regime(
        drawdown_stats,
        "Net Performance by Prior SPY Drawdown Regime",
        output / "11_market_drawdown_regime.png",
    )

    summary = {
        "candidate": winner,
        "data_scope": "research_snapshot_only",
        "pit_verified": False,
        "ic_horizon": "next_session_total_return",
        "average_completed_holding_period_sessions": float(periods.sessions.mean()),
        "dispersion_training_tertiles": {"low": float(low_cut), "high": float(high_cut)},
        "annual": annual.reset_index().to_dict(orient="records"),
        "ic_by_year": ic_year.reset_index(names="year").to_dict(orient="records"),
        "holding_period_by_start_year": holding.reset_index().to_dict(orient="records"),
        "long_short_pnl_percent_by_year": leg_year.reset_index(names="year").to_dict(orient="records"),
        "sector_neutralization_by_year": sector_year.reset_index(names="year").to_dict(orient="records"),
        "beta_neutralization_by_year": beta_year.reset_index(names="year").to_dict(orient="records"),
        "residual_volatility_bucket": bucket_summary.reset_index(names="bucket").to_dict(orient="records"),
        "dispersion_regime": dispersion_stats.reset_index().to_dict(orient="records"),
        "market_drawdown_regime": drawdown_stats.reset_index().to_dict(orient="records"),
    }
    (output / "diagnostics_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("output/equities_2026-09-20_sip_r2"))
    parser.add_argument("--quality", type=Path, default=Path("reports/equity_v2/data_quality"))
    parser.add_argument(
        "--selection", type=Path, default=Path("reports/equity_v2/backtest/selection_lock.json")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("reports/equity_final/diagnostics")
    )
    args = parser.parse_args()
    main(args.source, args.quality, args.selection, args.output)

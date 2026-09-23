"""Broader price-path alpha study with frozen train/validation selection."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd
from evaluate_equity_v2 import AUDIT, TRAIN, VALIDATION, read_panel, segment_metrics, sha256

from pairs_trading.equity_backtest import FastPortfolioConfig, residual_returns, run_fast_backtest
from pairs_trading.equity_portfolio import PortfolioCosts


def build_signals(
    adjusted_close: pd.DataFrame,
    adjusted_open: pd.DataFrame,
    residual: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    signals: dict[str, pd.DataFrame] = {
        "low_residual_vol_60": -residual.rolling(60, min_periods=60).std(),
        "low_residual_vol_126": -residual.rolling(126, min_periods=126).std(),
        "price_to_high_252": adjusted_close.div(
            adjusted_close.rolling(252, min_periods=252).max()
        ),
    }
    for window in (63, 126):
        path = residual.rolling(window, min_periods=window)
        signals[f"residual_trend_{window}"] = path.mean().div(path.std().where(path.std() > 0))
        signals[f"residual_efficiency_{window}"] = path.sum().div(
            residual.abs().rolling(window, min_periods=window).sum().where(lambda x: x > 0)
        )
    overnight = np.log(adjusted_open.div(adjusted_close.shift(1)))
    intraday = np.log(adjusted_close.div(adjusted_open))
    for window in (21, 63):
        signals[f"overnight_{window}"] = overnight.rolling(window, min_periods=window).sum()
        signals[f"intraday_{window}"] = intraday.rolling(window, min_periods=window).sum()
    signals["blend_high_defensive"] = (
        signals["price_to_high_252"].rank(axis=1, pct=True)
        + signals["low_residual_vol_60"].rank(axis=1, pct=True)
        - 1
    )
    return signals


def main(source: Path, v2_output: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    initial = pd.read_csv("reports/equity_v2/data_quality/prospective_400_stocks.csv")
    audit = pd.read_csv(v2_output / "backtest_universe_audit.csv")
    symbols = audit.loc[audit.included, "symbol"].tolist()
    universe = initial[initial.symbol.isin(symbols)].copy()
    sectors = universe.set_index("symbol").sector.reindex(symbols)
    calendar = pd.read_csv(source / "market_calendar.csv")
    sessions = pd.DatetimeIndex(pd.to_datetime(calendar.date))
    sessions = sessions[(sessions >= "2017-01-03") & (sessions <= AUDIT[1])]
    adjusted_close = read_panel(source / "bars_all.csv", symbols + ["SPY"], sessions, "c")
    adjusted_open = read_panel(source / "bars_all.csv", symbols, sessions, "o")
    raw_close = read_panel(source / "bars_raw.csv", symbols, sessions, "c")
    raw_volume = read_panel(source / "bars_raw.csv", symbols, sessions, "v")
    returns = adjusted_close[symbols].pct_change(fill_method=None)
    market = adjusted_close.SPY.pct_change(fill_method=None)
    residual, beta = residual_returns(returns, market, sectors)
    signals = build_signals(adjusted_close[symbols], adjusted_open, residual)
    candidates = []
    for signal_name, score in signals.items():
        for transform in ("rank", "tail20"):
            for frequency in (10, 21):
                name = f"{signal_name}_{transform}_r{frequency}"
                candidates.append((name, score, transform, frequency))
    protocol = Path("docs/EQUITY_RESEARCH_PROTOCOL_V4.md")
    costs = PortfolioCosts()
    base = FastPortfolioConfig()
    freeze = {
        "protocol_sha256": sha256(protocol),
        "v2_universe_audit_sha256": sha256(v2_output / "backtest_universe_audit.csv"),
        "adjusted_bars_sha256": sha256(source / "bars_all.csv"),
        "raw_bars_sha256": sha256(source / "bars_raw.csv"),
        "costs": asdict(costs),
        "base_config": asdict(base),
        "candidates": [name for name, *_ in candidates],
        "audit_status": "REUSED_HOLDOUT_NOT_PRISTINE",
    }
    (output / "frozen_run_inputs.json").write_text(json.dumps(freeze, indent=2) + "\n")
    development = (returns.index >= TRAIN[0]) & (returns.index <= VALIDATION[1])
    rows = []
    for name, score, transform, frequency in candidates:
        config = replace(base, score_transform="rank" if transform == "rank" else "tail",
                         tail_fraction=0.20, rebalance_every=frequency)
        daily, weights = run_fast_backtest(
            score.loc[development], returns.loc[development], beta.loc[development], sectors,
            raw_close.loc[development], raw_volume.loc[development], config=config, costs=costs,
        )
        train = segment_metrics(daily, weights, beta, *TRAIN)
        validation = segment_metrics(daily, weights, beta, *VALIDATION)
        robust = min(train["sharpe"], validation["sharpe"])
        mean = (train["sharpe"] + validation["sharpe"]) / 2
        eligible = (robust > 0 and validation["max_drawdown"] >= -0.25
                    and validation["average_abs_net"] <= 0.01
                    and validation["annual_turnover"] <= 80)
        rows.append({"candidate": name, "eligible": eligible, "robust_sharpe": robust,
                     "mean_sharpe": mean, **{f"train_{k}": v for k, v in train.items()},
                     **{f"validation_{k}": v for k, v in validation.items()}})
    table = pd.DataFrame(rows).sort_values(
        ["eligible", "robust_sharpe", "mean_sharpe", "validation_cagr", "candidate"],
        ascending=[False, False, False, False, True],
    )
    table.to_csv(output / "candidate_selection.csv", index=False)
    if not table.eligible.any():
        result = {"status": "REJECTED", "reason": "NO_CANDIDATE_PASSED_FROZEN_GATES",
                  "table_sha256": sha256(output / "candidate_selection.csv")}
        (output / "selection_lock.json").write_text(json.dumps(result, indent=2) + "\n")
        print(table.head(10).to_string(index=False))
        return
    winner = str(table.loc[table.eligible].iloc[0].candidate)
    lock = {"status": "SELECTED", "candidate": winner, "selected_before_reused_audit": True,
            "table_sha256": sha256(output / "candidate_selection.csv"),
            "protocol_sha256": sha256(protocol)}
    (output / "selection_lock.json").write_text(json.dumps(lock, indent=2) + "\n")
    _, score, transform, frequency = next(item for item in candidates if item[0] == winner)
    config = replace(base, score_transform="rank" if transform == "rank" else "tail",
                     tail_fraction=0.20, rebalance_every=frequency)
    study = returns.index >= TRAIN[0]
    daily, weights = run_fast_backtest(
        score.loc[study], returns.loc[study], beta.loc[study], sectors,
        raw_close.loc[study], raw_volume.loc[study], config=config, costs=costs,
    )
    metrics = {"candidate": winner, "audit_status": "REUSED_HOLDOUT_NOT_PRISTINE",
               "train": segment_metrics(daily, weights, beta, *TRAIN),
               "validation": segment_metrics(daily, weights, beta, *VALIDATION),
               "reused_audit": segment_metrics(daily, weights, beta, *AUDIT)}
    (output / "evaluation.json").write_text(json.dumps(metrics, indent=2) + "\n")
    daily.to_csv(output / "winner_daily.csv")
    weights.loc[AUDIT[0]:].to_csv(output / "winner_weights_reused_audit.csv.gz", compression="gzip")
    latest = weights.iloc[-1].rename("target_weight").rename_axis("symbol").reset_index()
    latest = latest.merge(universe[["symbol", "security_id", "sector"]], on="symbol", how="left")
    latest["order_authorized"] = False
    latest.to_csv(output / "paper_targets_preview.csv", index=False)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("output/equities_2026-09-20_sip_r2"))
    parser.add_argument("--v2-output", type=Path, default=Path("reports/equity_v2/backtest"))
    parser.add_argument("--output", type=Path, default=Path("reports/equity_v4"))
    args = parser.parse_args()
    main(args.source, args.v2_output, args.output)

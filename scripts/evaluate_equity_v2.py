"""Evaluate the frozen current-cohort equity grid and lock audit before reading it."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from pairs_trading.equity_backtest import (
    FastPortfolioConfig,
    candidate_scores,
    performance_metrics,
    residual_returns,
    run_fast_backtest,
)
from pairs_trading.equity_portfolio import PortfolioCosts

TRAIN = (pd.Timestamp("2018-01-02"), pd.Timestamp("2022-12-30"))
VALIDATION = (pd.Timestamp("2023-01-03"), pd.Timestamp("2024-12-31"))
AUDIT = (pd.Timestamp("2025-01-02"), pd.Timestamp("2026-09-18"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_panel(path: Path, symbols: list[str], sessions: pd.DatetimeIndex, field: str) -> pd.DataFrame:
    frame = pd.read_csv(path, usecols=["symbol", "t", field])
    frame = frame[frame.symbol.isin(symbols)]
    frame["session"] = (
        pd.to_datetime(frame.t, utc=True)
        .dt.tz_convert("America/New_York")
        .dt.tz_localize(None)
        .dt.normalize()
    )
    return frame.pivot(index="session", columns="symbol", values=field).reindex(
        index=sessions, columns=symbols
    )


def segment_metrics(
    daily: pd.DataFrame, weights: pd.DataFrame, beta: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp
) -> dict:
    section = daily.loc[start:end]
    section_weights = weights.loc[section.index]
    result = performance_metrics(section.net_return)
    years = len(section) / 252
    result.update(
        gross_total_return=performance_metrics(section.gross_return)["total_return"],
        total_transaction_cost=float(section.transaction_cost.sum()),
        total_borrow_cost=float(section.borrow_cost.sum()),
        annual_turnover=float(section.turnover.sum() / years) if years else np.nan,
        average_gross=float(section.gross.mean()),
        average_abs_net=float(section.net.abs().mean()),
        average_abs_beta=float((section_weights * beta.loc[section.index]).sum(axis=1).abs().mean()),
        missing_held_sessions=int(section.missing_held_return.sum()),
    )
    return result


def main(source: Path, quality: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    protocol = Path("docs/EQUITY_RESEARCH_PROTOCOL.md")
    universe = pd.read_csv(quality / "prospective_400_stocks.csv")
    universe = universe.sort_values("security_id")
    symbols = universe.symbol.tolist()
    sectors = universe.set_index("symbol").sector.reindex(symbols)
    calendar = pd.read_csv(source / "market_calendar.csv")
    sessions = pd.DatetimeIndex(pd.to_datetime(calendar.date))
    sessions = sessions[(sessions >= "2017-01-03") & (sessions <= AUDIT[1])]
    adjusted = read_panel(source / "bars_all.csv", symbols + ["SPY"], sessions, "c")
    raw_close = read_panel(source / "bars_raw.csv", symbols, sessions, "c")
    raw_volume = read_panel(source / "bars_raw.csv", symbols, sessions, "v")
    study_sessions = sessions[(sessions >= TRAIN[0]) & (sessions <= AUDIT[1])]
    complete = adjusted.loc[study_sessions, symbols].notna().all()
    preliminary_returns = adjusted[symbols].pct_change(fill_method=None)
    discontinuity = preliminary_returns.loc[study_sessions].abs().gt(0.50).any()
    exclusions = pd.DataFrame(
        {
            "symbol": symbols,
            "continuous_adjusted_close": complete.reindex(symbols).to_numpy(),
            "adjusted_move_over_50pct": discontinuity.reindex(symbols).to_numpy(),
        }
    )
    exclusions["included"] = exclusions.continuous_adjusted_close & ~exclusions.adjusted_move_over_50pct
    exclusions.to_csv(output / "backtest_universe_audit.csv", index=False)
    symbols = exclusions.loc[exclusions.included, "symbol"].tolist()
    if len(symbols) < 200:
        raise RuntimeError(f"only {len(symbols)} continuous symbols; at least 200 required")
    universe = universe[universe.symbol.isin(symbols)].copy()
    sectors = sectors.reindex(symbols)
    adjusted = adjusted[symbols + ["SPY"]]
    raw_close = raw_close[symbols]
    raw_volume = raw_volume[symbols]
    returns = adjusted[symbols].pct_change(fill_method=None)
    market = adjusted.SPY.pct_change(fill_method=None)

    # Keep every detected identifier-continuity exception in an explicit audit file.
    quarantine_rows = []
    for symbol in exclusions.loc[exclusions.adjusted_move_over_50pct, "symbol"]:
        boundaries = preliminary_returns.index[preliminary_returns[symbol].abs().gt(0.50)]
        if len(boundaries):
            for boundary in boundaries:
                quarantine_rows.append(
                    {"symbol": symbol, "boundary_session": boundary,
                     "return": preliminary_returns.at[boundary, symbol]}
                )
    pd.DataFrame(quarantine_rows).to_csv(output / "identifier_continuity_quarantine.csv", index=False)

    residual, beta = residual_returns(returns, market, sectors)
    scores = candidate_scores(residual)
    config = FastPortfolioConfig()
    costs = PortfolioCosts()
    hashes = {
        "protocol": sha256(protocol),
        "universe": sha256(quality / "prospective_400_stocks.csv"),
        "adjusted_bars": sha256(source / "bars_all.csv"),
        "raw_bars": sha256(source / "bars_raw.csv"),
    }
    (output / "frozen_run_inputs.json").write_text(
        json.dumps(
            {
                "hashes": hashes,
                "config": asdict(config),
                "costs": asdict(costs),
                "segments": {
                    "train": [str(x.date()) for x in TRAIN],
                    "validation": [str(x.date()) for x in VALIDATION],
                    "audit": [str(x.date()) for x in AUDIT],
                },
                "candidate_names": sorted(scores),
                "data_scope": "research_snapshot_only",
                "pit_verified": False,
            },
            indent=2,
        )
        + "\n"
    )

    runs: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    selection_rows = []
    # Run only through validation while selecting; no audit metrics are evaluated here.
    through_validation = (returns.index >= TRAIN[0]) & (returns.index <= VALIDATION[1])
    for name in sorted(scores):
        daily, weights = run_fast_backtest(
            scores[name].loc[through_validation],
            returns.loc[through_validation],
            beta.loc[through_validation],
            sectors,
            raw_close.loc[through_validation],
            raw_volume.loc[through_validation],
            config=config,
            costs=costs,
        )
        train = segment_metrics(daily, weights, beta, *TRAIN)
        validation = segment_metrics(daily, weights, beta, *VALIDATION)
        eligible = (
            validation["sessions"] >= 100
            and validation["max_drawdown"] >= -0.25
            and validation["average_abs_net"] <= 0.01
            and validation["annual_turnover"] <= 80
            and validation["missing_held_sessions"] == 0
        )
        selection_rows.append(
            {"candidate": name, "eligible": eligible, **{f"train_{k}": v for k, v in train.items()},
             **{f"validation_{k}": v for k, v in validation.items()}}
        )
        runs[name] = (daily, weights)
    selection = pd.DataFrame(selection_rows).sort_values(
        ["eligible", "validation_sharpe", "validation_cagr", "train_sharpe", "candidate"],
        ascending=[False, False, False, False, True],
    )
    selection.to_csv(output / "candidate_selection.csv", index=False)
    if not selection.eligible.any():
        raise RuntimeError("no candidate passed the frozen validation gates")
    winner = str(selection.loc[selection.eligible].iloc[0].candidate)
    lock = {
        "selected_candidate": winner,
        "selected_before_audit": True,
        "selection_table_sha256": sha256(output / "candidate_selection.csv"),
        "protocol_sha256": hashes["protocol"],
        "selection_rule": "validation_sharpe_then_validation_cagr_then_train_sharpe_then_name",
    }
    (output / "selection_lock.json").write_text(json.dumps(lock, indent=2) + "\n")

    # Only after the lock exists do we run the winner through the audit segment.
    study = returns.index >= TRAIN[0]
    daily, weights = run_fast_backtest(
        scores[winner].loc[study], returns.loc[study], beta.loc[study], sectors,
        raw_close.loc[study], raw_volume.loc[study], config=config, costs=costs
    )
    metrics = {
        "candidate": winner,
        "data_scope": "research_snapshot_only",
        "pit_verified": False,
        "train": segment_metrics(daily, weights, beta, *TRAIN),
        "validation": segment_metrics(daily, weights, beta, *VALIDATION),
        "audit": segment_metrics(daily, weights, beta, *AUDIT),
    }
    (output / "evaluation.json").write_text(json.dumps(metrics, indent=2, allow_nan=False) + "\n")
    daily.to_csv(output / "winner_daily.csv")
    weights.loc[AUDIT[0] :].to_csv(output / "winner_weights_audit.csv.gz", compression="gzip")
    latest = weights.iloc[-1].rename("target_weight").rename_axis("symbol").reset_index()
    latest = latest.merge(universe[["symbol", "security_id", "sector"]], on="symbol", how="left")
    latest["order_authorized"] = False
    latest.to_csv(output / "paper_targets_preview.csv", index=False)
    print(json.dumps(metrics, indent=2, allow_nan=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("output/equities_2026-09-20_sip_r2"))
    parser.add_argument("--quality", type=Path, default=Path("reports/equity_v2/data_quality"))
    parser.add_argument("--output", type=Path, default=Path("reports/equity_v2/backtest"))
    args = parser.parse_args()
    main(args.source, args.quality, args.output)

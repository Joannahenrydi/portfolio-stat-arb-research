"""Collect and archive the frozen v12 ETF panel from Yahoo Finance."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yfinance as yf

UNIVERSE = [
    "SPY", "QQQ", "IWM", "EFA", "EEM", "VNQ", "SHY", "IEF", "TLT", "TIP",
    "LQD", "HYG", "GLD", "SLV", "DBC", "USO", "UNG", "DBA", "UUP",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect(output: Path, start: str, end: str) -> None:
    output.mkdir(parents=True, exist_ok=False)
    captured = datetime.now(timezone.utc).isoformat()
    data = yf.download(
        UNIVERSE, start=start, end=end, auto_adjust=False, actions=True,
        group_by="ticker", threads=True, progress=False,
    )
    rows = []
    for symbol in UNIVERSE:
        if symbol not in data.columns.get_level_values(0):
            raise RuntimeError(f"missing ETF response: {symbol}")
        frame = data[symbol].copy().reset_index()
        frame.columns = [str(column).lower().replace(" ", "_") for column in frame.columns]
        frame.insert(0, "symbol", symbol)
        rows.append(frame)
    panel = pd.concat(rows, ignore_index=True)
    panel["date"] = pd.to_datetime(panel.date).dt.tz_localize(None)
    required = ["date", "open", "high", "low", "close", "adj_close", "volume"]
    if panel[required].isna().all(axis=1).any():
        raise ValueError("provider returned an all-missing bar row")
    path = output / "etf_daily.csv"
    panel.to_csv(path, index=False)
    coverage = panel.groupby("symbol").agg(
        first=("date", "min"), last=("date", "max"),
        adjusted_observations=("adj_close", "count"), raw_observations=("close", "count"),
    ).reindex(UNIVERSE)
    coverage.to_csv(output / "coverage.csv")
    manifest = {
        "status": "downloaded_pending_quality_and_strategy_gates",
        "captured_at": captured,
        "source": "Yahoo Finance via yfinance",
        "scope": "research_third_party_adjusted",
        "start": start,
        "end_exclusive": end,
        "symbols": UNIVERSE,
        "rows": len(panel),
        "sha256": sha256(path),
        "limitations": [
            "Third-party adjusted history is not an exchange-grade PIT archive.",
            "Accepted research requires Alpaca/SIP replication before paper orders.",
            "Historical ETF borrow is unavailable; a frozen common liquid-ETF proxy is modeled.",
        ],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"rows": len(panel), "symbols": len(coverage), "sha256": manifest["sha256"]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start", default="2007-01-01")
    parser.add_argument("--end", default="2025-01-01")
    args = parser.parse_args()
    collect(args.output, args.start, args.end)

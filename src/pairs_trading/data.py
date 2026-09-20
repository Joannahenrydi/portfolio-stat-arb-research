"""Read-only historical daily data; never connects to a trading account."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


def download_daily(symbols: list[str], start: str, end: str, output: str) -> pd.DataFrame:
    import yfinance as yf

    if len(set(symbols)) != len(symbols) or len(symbols) < 2:
        raise ValueError("provide at least two distinct symbols")
    if pd.Timestamp(start) >= pd.Timestamp(end):
        raise ValueError("start must precede end (end is exclusive)")
    path = Path(output)
    path.mkdir(parents=True, exist_ok=True)
    # Keep the library's writable cache inside the requested artifact directory.
    yf.set_tz_cache_location(str(path / "yfinance_cache"))
    series, counts = [], {}
    for number, symbol in enumerate(symbols):
        raw = yf.Ticker(symbol).history(
            start=start, end=end, interval="1d", auto_adjust=False,
            actions=True, raise_errors=True,
        )
        if raw.empty or "Adj Close" not in raw:
            raise ValueError(f"missing adjusted daily prices for {symbol}")
        raw.to_csv(path / f"raw_{number}.csv")
        close = raw["Adj Close"].rename(symbol)
        close.index = pd.DatetimeIndex(close.index).tz_localize(None).normalize()
        if close.index.has_duplicates:
            raise ValueError(f"duplicate daily timestamps for {symbol}")
        series.append(close)
        counts[symbol] = len(close)
    joined = pd.concat(series, axis=1).sort_index()
    prices = joined.dropna()
    if prices.empty or not np.isfinite(prices.to_numpy()).all() or (prices <= 0).any().any():
        raise ValueError("invalid or empty aligned prices")
    prices.index.name = "date"
    prices.to_csv(path / "prices.csv")
    metadata = {
        "provider": "Yahoo Finance via yfinance (unofficial research interface)",
        "yfinance_version": yf.__version__,
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "symbols": symbols, "requested_start": start, "requested_end_exclusive": end,
        "price_field": "Adj Close", "interval": "1d", "source_rows": counts,
        "aligned_rows": len(prices), "dropped_noncommon_dates": len(joined) - len(prices),
        "first_date": str(prices.index[0].date()), "last_date": str(prices.index[-1].date()),
        "sha256": hashlib.sha256((path / "prices.csv").read_bytes()).hexdigest(),
        "limitations": [
            "Adjusted prices are total-return research proxies, not executable quotes.",
            "Current adjustment factors are not point-in-time corporate-action data.",
            "Daily-date alignment does not establish simultaneous cross-market closes.",
        ],
    }
    (path / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return prices

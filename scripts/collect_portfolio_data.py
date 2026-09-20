"""Collect auditable prototype data without changing existing research artifacts."""

import argparse
import json
from pathlib import Path

from pairs_trading.alpaca_data import download_alpaca
from pairs_trading.data import download_daily

p = argparse.ArgumentParser()
p.add_argument("--provider", choices=["yahoo", "alpaca"], default="yahoo")
p.add_argument("--feed", choices=["iex", "sip"], default="iex")
p.add_argument("--output", default="output/portfolio_v1/market_data")
p.add_argument("--config", default="config/portfolio_v1.json")
a = p.parse_args()
c = json.loads(Path(a.config).read_text())
symbols = list(c["groups"]) + [c["market"]]
if a.provider == "yahoo":
    download_daily(symbols, c["start"], c["end_exclusive"], a.output)
else:
    download_alpaca(symbols, c["start"], c["end_exclusive"], a.output, feed=a.feed)
print(f"Collected {len(symbols)} symbols into {a.output}; research only.")

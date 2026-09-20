import argparse
import json
from pathlib import Path

from pairs_trading.portfolio_research import write_research

p = argparse.ArgumentParser()
p.add_argument("--data", default="output/portfolio_v1/market_data")
p.add_argument("--output", default="reports/portfolio_2026-09-20")
a = p.parse_args()
c = json.loads(Path("config/portfolio_v1.json").read_text())
r = write_research(a.data, a.output, c, "research_protocol.md")
print(
    json.dumps(
        {"status": r["status"], "blend": r["metrics"]["blend"], "stress_pass": r["stress_pass"]},
        indent=2,
    )
)

# Portfolio Statistical Arbitrage Research

Daily US ETF statistical-arbitrage research with residual reversal, Kalman relative-value and volume-dislocation signals, market/group neutralization, portfolio constraints, transaction costs and temporal validation.

**Current decision: REJECTED — approved allocation remains cash. No orders submitted.** This project records failed hypotheses rather than claiming deployable alpha.

## Latest research

23 ETF/benchmark symbols were collected from Yahoo and authenticated Alpaca IEX through September 18, 2026. Three alpha families and their fixed equal-weight blend were evaluated with rolling historical parameter estimates, train/validation/audit time splits and seven stress scenarios.

| Fixed blend, net return | Validation 2023–2024 | Temporal audit 2025–2026 |
|---|---:|---:|
| Yahoo | -9.67% | -7.75% |
| Alpaca IEX | -2.08% | +1.85% |

Neither source passed the predefined acceptance criteria. The audit interval was previously inspected in the predecessor project and is **not pristine blind OOS**. IEX history is incomplete and volume represents one exchange. Costs and fills are proxies, not observed execution.

- [Research report](reports/portfolio_2026-09-20/REPORT.md)
- [Alpaca cross-check](reports/portfolio_2026-09-20/alpaca_iex/REPORT.md)
- [Coverage and source discrepancies](reports/portfolio_2026-09-20/DATA_AUDIT.md)
- [Frozen protocol](research_protocol.md)
- [Reproduction and outstanding work](docs/PORTFOLIO_RESEARCH.md)

## Run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[data,dev]'
python scripts/collect_portfolio_data.py --provider yahoo
python scripts/evaluate_portfolio.py
pytest
```

For Alpaca, set `APCA_API_KEY_ID` and `APCA_API_SECRET_KEY` in your local process environment, then use:

```bash
python scripts/collect_portfolio_data.py --provider alpaca --feed iex --output output/portfolio_v1/alpaca_data
python scripts/evaluate_portfolio.py --data output/portfolio_v1/alpaca_data --output reports/portfolio_2026-09-20/alpaca_iex
```

Credentials, raw vendor data and derived price panels are not committed. Download your own data to rerun the pipeline. Included reports contain aggregate research results, positions, attribution and source-quality diagnostics; hashes identify the original local snapshots.

## Structure

- `src/pairs_trading/portfolio_research.py`: features, portfolio construction, costs and validation.
- `src/pairs_trading/`: retained predecessor data adapters and pair research modules.
- `scripts/`: data collection and research entry points.
- `tests/`: offline tests for no-look-ahead, neutrality, costs, accounting and existing pair components.
- `config/`: frozen ETF universe and settings.
- `reports/`: research outputs and explicit rejection decisions.

The package namespace remains `pairs_trading` for compatibility with retained components. The inherited paper broker is not connected to the new portfolio pipeline; portfolio paper output is shadow targets only.


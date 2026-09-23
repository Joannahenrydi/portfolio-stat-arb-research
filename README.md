# Portfolio Statistical Arbitrage Research

Daily US equity statistical-arbitrage research with 356 continuously observed stocks from a liquid 400-stock current cohort, residual/Kalman/price-path alphas, market/sector neutralization, portfolio constraints, transaction costs and temporal validation.


## Latest equity research

Alpaca SIP raw and adjusted daily bars were collected through September 18, 2026 for 503 current candidates plus SPY. A liquid 400-name screen and a 356-name continuous-history research cohort were evaluated across 111 frozen candidates. The cohort is explicitly survivor-biased at the user's direction.

| Locked v2 winner, net | Train 2018–2022 | Validation 2023–2024 | Reused audit 2025–2026 |
|---|---:|---:|---:|
| CAGR | -1.82% | +0.37% | +6.07% |
| Sharpe | -0.30 | 0.11 | 0.83 |

The strategy failed the development gates. The audit interval is **not pristine blind OOS**. Costs and fills remain modeled rather than observed execution.

- [12-week equity execution report](reports/equity_final/REPORT.md)
- [Equity data audit](reports/equity_v2/data_quality/DATA_STATUS.md)
- [v2 frozen protocol](docs/EQUITY_RESEARCH_PROTOCOL.md)
- [v2 backtest](reports/equity_v2/backtest/evaluation.json)
- [Kill tests](reports/equity_v2/robustness/kill_tests.csv)
- [Promotion decision](reports/equity_final/promotion_decision.json)
- [Legacy ETF research report](reports/portfolio_2026-09-20/REPORT.md)
- [Alpaca cross-check](reports/portfolio_2026-09-20/alpaca_iex/REPORT.md)
- [Coverage and source discrepancies](reports/portfolio_2026-09-20/DATA_AUDIT.md)
- [Frozen protocol](research_protocol.md)
- [Reproduction and outstanding work](docs/PORTFOLIO_RESEARCH.md)

## Run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[alpaca,data,dev]'
python scripts/audit_equity_snapshot.py
python scripts/evaluate_equity_v2.py
python scripts/evaluate_equity_v3.py
python scripts/evaluate_equity_v4.py
python scripts/robustness_equity_v2.py
python scripts/generate_shadow_paper.py
pytest
```

For a new Alpaca collection, set `APCA_API_KEY_ID` and `APCA_API_SECRET_KEY` in your local process environment. Never put credentials in a file or commit. Collection scripts default to new local output directories; existing audited snapshots should remain immutable.

Credentials, raw vendor data and derived price panels are not committed. Included reports contain aggregate research results, positions, diagnostics and input hashes.

## Structure

- `src/pairs_trading/equity_alphas.py`: causal residual, Kalman and dislocation features.
- `src/pairs_trading/equity_backtest.py`: fast neutral targets, weight drift and cost accounting.
- `src/pairs_trading/equity_portfolio.py`: labeled constrained optimizer and cost estimator.
- `src/pairs_trading/pit_universe.py`: bitemporal universe evidence and data gates.
- `src/pairs_trading/`: retained predecessor data adapters and pair research modules.
- `scripts/`: SIP collection, audit, frozen research rounds, kill tests and shadow paper.
- `tests/`: offline tests for no-look-ahead, neutrality, costs, accounting and existing pair components.
- `config/`: frozen ETF universe and settings.
- `reports/`: research outputs and explicit rejection decisions.

The package namespace remains `pairs_trading` for compatibility with retained components. Portfolio paper output is shadow-only while the promotion decision is rejected.

## Origin and license

Extracted from [kalman-pairs-trading](https://github.com/Joannahenrydi/kalman-pairs-trading), preserving its Python namespace and pair-model tests. MIT license for code; market data remains subject to provider terms.

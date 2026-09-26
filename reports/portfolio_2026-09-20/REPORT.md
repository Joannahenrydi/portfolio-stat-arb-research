# Portfolio Strategy Research Result

**Status: REJECTED / CASH. No orders were submitted.**

Data source: Yahoo ETF prototype.

The panel contains 55,776 daily bars for 23 symbols from 2017-01-03 through 2026-09-18: 22 candidate
ETFs plus SPY as benchmark.

| Period | After-cost return | Sharpe | Max drawdown | Average daily turnover |
|---|---:|---:|---:|---:|
| Validation 2023–2024 | -9.67% | -19.99 | -9.67% | 48.21% |
| Holdout audit 2025–2026 | -7.75% | -14.73 | -7.76% | 48.27% |

The frozen portfolio equally blends residual reversal, Kalman dynamic relative value and
price/volume dislocation. It neutralizes ETF economic groups and market beta, models holding drift,
executes at the next close, and charges impact, spread, commission and borrow.

Base statistical gates passed: False. All stress tests passed: False. See `evaluation.json` for
sleeve, annual and cost attribution; CSV files contain daily returns, positions and name-level PnL.

## Stress tests

| Scenario | Validation return | Audit return |
|---|---:|---:|
| double_cost | -18.56% | -15.62% |
| delay_one_more | -9.97% | -8.83% |
| horizon_2 | -9.66% | -7.72% |
| horizon_5 | -9.94% | -7.90% |
| alternate_days | -5.67% | -4.79% |
| remove_top_5pct | -9.52% | -7.50% |
| remove_20pct | -8.94% | -7.06% |

## Outstanding production requirements

- Historical PIT security master and delisting reconciliation are unavailable.
- Prices are retrospectively adjusted; raw execution and action reconciliation is incomplete.
- Borrow availability and observed spreads/fills are unavailable.
- This is an ETF prototype; the 200–500-stock universe was not collected for this run.
- The 2025–2026 interval was inspected in a previous project and is not blind OOS.

This ETF prototype is not a PIT backtest of 400 U.S. equities. IEX volume cannot represent total
market ADV, and results are not directly comparable with consolidated-market data. Costs are frozen
proxies and do not establish tradable alpha. Missing market cap and historical sector labels were
not fabricated. `paper_targets.csv` contains research_weight for observation only; every
approved_weight is zero. Production requires historical security-master and corporate-action data,
reconciled paper fills and two to three months of prospective observation.

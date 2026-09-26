# 12-Week Equity Statistical-Arbitrage Research Report

## Decision

**REJECTED / CASH: orders are disabled and approved exposure is zero.**

The system completed equity data collection, alpha generation, market/sector/beta neutralization,
portfolio construction, cost modeling, chronological evaluation, stress tests and shadow-target
generation. No after-cost alpha remained consistently positive in both train and validation.
Backtest performance therefore cannot be promised and paper orders cannot begin. Positive returns
in 2025–2026 are reused holdout evidence and do not override weaker train/validation results.

## Data and universe

- Alpaca SIP: 503 current equity candidates plus SPY, from 2017-01-03 through 2026-09-18.
- 1,161,734 raw and 1,161,734 adjusted daily bars; zero key mismatches, duplicates or invalid OHLCV rows.
- The initial screen admitted 499 stocks and selected the top 400 by liquidity. To avoid treating
  missing held returns as zero, historical backtests used the 356 stocks with adjusted closes on
  every 2018–2026 session and no research-period identity discontinuity above 50%.
- The archive contains 14,710 corporate-action records and an independent trading calendar; 111 of
  117 requested historical IVV monthly holdings snapshots were accepted.
- Later user direction permitted current-constituent backfilling, so results are explicitly labeled
  `research_snapshot_only` and retain material survivor, complete-history and current-sector bias.

## Alpha and portfolio

Implemented families include short-horizon residual reversal, Kalman prior innovation,
price/volume and volatility dislocation, raw and residual momentum, low idiosyncratic volatility,
trend efficiency, distance from the 52-week high, and overnight/intraday momentum. At every
rebalance, scores are projected away from dollar net exposure, current-sector dummies and prior
126-session beta, then constrained by gross, single-name, turnover and ADV participation limits.
Holdings drift with realized returns between rebalances; the backtest does not assume free daily
rebalancing.

Costs include 0.5 bps commission, 2 bps half-spread, 1 bp slippage, square-root impact and 3% annual
borrow. Signals form at close `t` and can first earn returns from `t` to `t+1`.

The winning signal was separately calibrated on 379,947 completed 2018–2022 labels. A shrinkage
covariance estimate used 252 sessions ending before 2026-09-18, and the cost-aware optimizer ran on
all 356 stocks. Its optimum was `NO_ECONOMIC_TRADE`: expected alpha did not cover costs, so the
theoretical target was all cash. The artifact in `optimized_portfolio/` does not change the rejection.

## Chronological results

Among nine pre-frozen v2 candidates, validation Sharpe selected 252-session residual momentum with
the latest 21 sessions skipped:

| Period | Net CAGR | Sharpe | Max drawdown | Annual turnover |
|---|---:|---:|---:|---:|
| Train 2018–2022 | -1.82% | -0.30 | -11.94% | 6.94x |
| Validation 2023–2024 | 0.37% | 0.11 | -4.53% | 5.65x |
| Reused audit 2025–2026-09 | 6.07% | 0.83 | -5.32% | 8.67x |

The subsequently frozen v3 matrix contained 54 momentum concentration/rebalance candidates, and
v4 contained 48 alternative price-path alpha candidates. Neither study produced a candidate with
positive Sharpe in both train and validation. Across 111 frozen candidates, none passed the robust
promotion rules.

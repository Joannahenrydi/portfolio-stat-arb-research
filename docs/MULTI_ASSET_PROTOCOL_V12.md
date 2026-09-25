# v12 - ETF Cross-Asset Relative Value

Frozen on 2026-09-25 before ETF data were downloaded or any v12 signal/result was evaluated.

## Objective and evidence

v12 is a new strategy family, not another equity residual-reversal variant. It combines
asset-specific trend and cross-asset relative-value signals across liquid US-listed ETFs, then
controls dollar, equity, duration, credit, commodity and USD factor exposure at portfolio level.

- Train: 2008-01-02 through 2016-12-30.
- Validation/model selection: 2017-01-03 through 2020-12-31.
- Locked test: 2021-01-04 through 2024-12-31, loaded only after selection and kill-test locks.
- Prospective shadow: first new session after the final accepted specification is committed.

Yahoo Finance adjusted OHLCV is used only because Alpaca credentials are not configured in the
runtime. It is labeled `research_third_party_adjusted`; an accepted historical candidate remains
pending Alpaca/SIP replication before paper orders.

## Frozen ETF universe

| Sleeve | ETFs |
|---|---|
| US/global equity | SPY, QQQ, IWM, EFA, EEM, VNQ |
| Treasury/inflation | SHY, IEF, TLT, TIP |
| Credit | LQD, HYG |
| Precious metals | GLD, SLV |
| Broad/energy/agriculture commodities | DBC, USO, UNG, DBA |
| USD | UUP |

A date requires adjusted close, raw volume, 252 prior returns and prior 60-session median dollar
volume. Missing data are never filled across an ETF's inception.

## Alpha families

1. `trend`: equal standardized 3-1, 6-1 and 12-1 month total-return momentum, divided by prior
   60-session volatility.
2. `kalman_relative_value`: dynamic-beta standardized innovations for the frozen pairs SPY-TLT,
   QQQ-IWM, HYG-LQD, HYG-SPY, GLD-TIP, SLV-GLD, DBC-UUP, USO-DBC, and LQD-IEF. Pair signals are
   mapped as equal and opposite expected returns to their two legs.
3. `ratio_mean_reversion`: negative 252-session z-score of log price ratios for the same pairs,
   mapped equally and oppositely.

Each family is calibrated once on train to next-five-session returns with a positive
zero-intercept ridge slope. A nonpositive slope blocks that family; its sign cannot be flipped from
validation or test.

## Portfolio construction

At five-session rebalances, expected returns feed the existing covariance-aware optimizer. The
covariance is a prior 126-session estimate with 50% shrinkage to its diagonal. Constraints are:

- gross at most 1.0, single ETF at most 15%, turnover at most 25% per rebalance;
- entry participation at most 10 bps of prior ADV for USD 100,000 NAV;
- exact dollar neutrality and exact zero exposure to the fixed equity, duration, credit,
  commodity and USD columns whenever the exposure system has rank;
- 0 commission, 1 bp half-spread, 1 bp slippage, square-root impact 0.10, and 1% annual borrow for
  liquid ETF shorts.

Frozen factor loadings are engineering risk coordinates, not estimated betas. They are stored in
the run artifact before backtesting.

## Frozen experiment matrix

| ID | Alpha | Selectable |
|---|---|---|
| M00 | trend | yes |
| M01 | Kalman relative value | conditional |
| M02 | ratio mean reversion | conditional |
| M03 | equal blend of admitted families | conditional |
| M04 | train-ICIR blend of admitted families | conditional |
| M05 | M04 with 10-session rebalance | conditional |

## Gates

A candidate needs train net Sharpe above 0.70, validation net Sharpe above 0.50, validation net
CAGR above 5%, drawdown no worse than -15% in both intervals, annual turnover no greater than 25x,
zero constraint failures, and positive validation return with 2x costs, one-session signal delay,
one pair removed at a time, and rebalance frequencies 3 and 10. Only after both locks are durable
may the 2021-2024 test be loaded. Test Sharpe must exceed 0.50, test CAGR 5%, and test drawdown must
remain above -15%. Failure leaves the strategy rejected.


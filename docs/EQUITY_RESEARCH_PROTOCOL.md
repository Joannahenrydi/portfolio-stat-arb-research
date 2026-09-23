# Equity research protocol v2

Frozen before inspecting any equity-strategy validation or audit result on 2026-09-23.

## Data scope

- SIP daily bars, adjusted for signal/return measurement and raw for liquidity.
- Start from the current liquid 400-stock cohort captured on 2026-09-20. Require a
  continuous adjusted close on every study session from 2018-01-02 through 2026-09-18;
  356 names pass. This is explicitly a `research_snapshot_only` universe and has severe
  survivorship, complete-history, and historical-sector bias.
- A symbol with an adjusted one-day move whose magnitude exceeds 50% during the study is
  excluded as an identifier-continuity exception; the event remains in the audit file.
- Signals observed at close `t` can first earn the close-to-close return from `t` to
  `t+1`. Missing prices never become zero returns.

## Locked time split

- Training: 2018-01-02 through 2022-12-30.
- Validation: 2023-01-03 through 2024-12-31.
- Audit: 2025-01-02 through 2026-09-18.
- Candidate selection uses training and validation only. The audit segment is run once
  after the selected candidate is written to the selection artifact.

## Candidate grid

All candidates use market- and sector-residual returns and rebalance every five market
sessions. The finite grid is:

- reversal: 1, 3, 5, and 10 sessions;
- residual momentum: 21 sessions skipping 5, 63 skipping 5, 126 skipping 21, and
  252 skipping 21;
- equal blend of 3-session reversal and 63-session momentum skipping 5.

Scores are cross-sectionally ranked, projected off an intercept, current sector dummies,
and a 126-session prior beta, then scaled to 100% gross. Limits are 1% per stock, 25%
one-way-plus-return turnover per rebalance, and 10 bps of trailing 60-session median raw
dollar volume. Portfolio NAV for capacity checks is $100,000.

## Costs and selection

Each trade pays 0.5 bps commission, 2 bps half-spread, 1 bp slippage, and square-root
impact coefficient 0.10 times trailing volatility and participation. Shorts pay 3% annual
borrow on calendar days. Missing ADV blocks a trade.

Rank candidates by validation net Sharpe, subject to at least 100 invested validation
sessions, validation maximum drawdown no worse than 25%, absolute average drifted net
exposure below 1% of NAV, and validation annual turnover below 80 times NAV. Targets are
neutral at each rebalance; between rebalances their marked weights are allowed to drift.
Ties use validation net
CAGR, then training net Sharpe, then candidate name. Report every candidate, including
failures. No audit result may change the winner.

## Status

This protocol measures a biased current-cohort research backtest because the user waived
the historical-membership requirement. It does not certify future returns and does not
authorize orders. Paper trading requires a separate prospective run and live checks.

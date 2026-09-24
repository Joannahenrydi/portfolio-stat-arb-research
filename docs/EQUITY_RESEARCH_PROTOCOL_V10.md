# v10 - Expanded Tail-Selective Adaptive Kalman

Frozen on 2026-09-25 before any v10 candidate was evaluated.

## Objective

v9 showed positive gross adaptive-Kalman alpha but insufficient edge per unit turnover and short
borrow. v10 tests two structural changes without changing the evidence gates: expand the daily
opportunity set from the earlier 341-name complete-history cohort, and concentrate gross exposure
in cross-sectional tails while slowing rebalance frequency. It does not relax costs, neutrality,
or acceptance thresholds to manufacture an acceptance.

Train is 2018-01-02 through 2022-12-30. Development is 2023-01-03 through 2024-12-31.
2025-01-02 through 2026-09-18 remains reused audit and may be loaded only after selection and
kill-test locks. Prospective evidence begins after this freeze.

## Expanded universe

The source is the frozen 503-name Alpaca current S&P 500 snapshot plus SPY. It remains
`research_snapshot_only`; current membership and sector labels are not historical PIT evidence.
Unlike v9, v10 does not require a security to have complete history from 2018. A name becomes
eligible only after 252 observed daily returns. On every decision date it must also have:

- raw close at least USD 5;
- positive current raw volume and a finite return;
- prior 60-session median dollar volume at least USD 10 million;
- a rank within the top 400 by that prior ADV;
- a nonmissing sector, prior beta, residual volatility, and alpha.

The rules use only information available at the decision close. Missing pre-IPO history remains
missing. No present-day shortable flag is projected backward.

## Signal and construction

The signal is the causal v9 R-adaptive Kalman innovation: fixed `Q=1e-5`, lagged EWMA R with
half-life 20, and 60-observation warm-up. The negative standardized innovation has its
contemporaneous leave-one-out sector mean removed and is then cross-sectionally standardized.

Tail selection is deterministic. For tail fraction `q`, retain scores with daily percentile
`p <= q` or `p > 1-q`; middle scores are ineligible and existing middle positions must be
liquidated through the same cost ledger. The smoothed variant applies a causal EWMA with half-life
3 sessions before tail selection and remasks current eligibility. No threshold is fitted.

Every target uses the v7 strict projection and must satisfy dollar, prior market-beta, and every
sector exposure within `1e-8`. Costs remain 0.5 bp commission, 2 bp half-spread, 1 bp slippage,
square-root impact coefficient 0.10, and 3% annual common borrow. Limits remain 1.0 gross, 1% per
name, 25% turnover per rebalance, 10 bps ADV participation, and a 95% target buffer.

## Frozen matrix

| ID | Universe / signal / rebalance | Selectable |
|---|---|---|
| X00 | v9 341-name K01 replay, every 3 sessions | reference only |
| X01 | expanded top-400, all scores, every 3 sessions | yes |
| X02 | expanded, outer 25% per side, every 3 sessions | yes |
| X03 | expanded, outer 15% per side, every 3 sessions | yes |
| X04 | expanded, outer 25% per side, every 5 sessions | yes |
| X05 | expanded, outer 15% per side, every 5 sessions | yes |
| X06 | expanded, outer 25% per side, every 10 sessions | yes |
| X07 | expanded, outer 15% per side, every 10 sessions | yes |
| X08 | expanded, EWMA-3 score, outer 15% per side, every 5 sessions | yes |

## Acceptance and kills

A candidate must have train net Sharpe above 0.70, development net Sharpe above 0.50,
development net CAGR above 5%, drawdown no worse than -15% in both intervals, annual turnover no
greater than 25x, zero strict-neutrality failures, and at least 100 average eligible names in
development. Rank survivors by lower train/development Sharpe, worse calendar-year return,
development CAGR, lower turnover, then ID.

The winner must retain positive development Sharpe with 2x costs and positive development total
return after a one-session signal delay, deterministic removal of 20% of names, removal of the top
5% PnL contributors, tail fraction perturbed by +/-20%, and neighboring rebalance intervals. A
failure rejects v10. The reused audit cannot rescue a rejection.

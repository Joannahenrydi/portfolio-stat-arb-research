# v13 - Cross-Asset Trend Risk-Budget Experiment

Frozen on 2026-09-25 before any v13 candidate was evaluated.

v12 showed positive train information in ETF trend but forced every macro exposure to exactly zero.
v13 is a controlled portfolio-construction experiment: it freezes the v12 trend score, calibration,
data, periods and costs, then changes only the exposure constraint strength. It does not introduce a
new alpha feature or tune a signal parameter after observing validation.

## Frozen alpha

The only alpha is the v12 `trend` family: the cross-sectional z-score of the average
volatility-normalized 3-1, 6-1 and 12-1 month ETF momentum signal. Its zero-intercept slope is fit
once on 2008-2016 against next-five-session returns. A nonpositive training slope blocks v13.

## Risk-budget optimizer

At each five-session rebalance, the risk-budget optimizer maximizes the frozen calibrated expected
return minus a smooth one-way trading-cost and projected weekly borrow penalty. All variants retain:

- gross <= 1.0, absolute single ETF <= 15%, turnover <= 25%;
- forecast annual volatility <= 10% using a prior 126-session covariance with 50% diagonal shrink;
- each trade <=10 bps of prior ADV at USD 100,000 NAV.

The factor caps control unintended macro bets; they do not require the intended trend exposure to
vanish. Realized costs remain 0 commission, 1 bp half-spread, 1 bp slippage, square-root impact
0.10 and 1% annual common liquid-ETF borrow.

## Frozen constraint matrix

All factor loadings are normalized to a maximum absolute loading of one before applying caps.

| ID | Constraint regime | Dollar net | Equity | Duration | Credit | Commodity | USD | Selectable |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| B00 | exact-neutral v12 replay | 0 | 0 | 0 | 0 | 0 | 0 | reference only |
| B01 | tight risk budgets | 10% | 10% | 10% | 5% | 10% | 8% | yes |
| B02 | medium risk budgets | 35% | 25% | 25% | 15% | 25% | 20% | yes |
| B03 | loose risk budgets | 60% | 50% | 50% | 30% | 50% | 40% | yes |

The corresponding sleeve gross caps are:

| Regime | Equity | Rates | Credit | Metals | Commodity | USD |
|---|---:|---:|---:|---:|---:|---:|
| Tight | 20% | 20% | 15% | 15% | 20% | 10% |
| Medium | 40% | 40% | 30% | 30% | 40% | 15% |
| Loose | 60% | 60% | 40% | 40% | 60% | 15% |

Train/validation/test splits and acceptance gates are identical to v12. B00 diagnoses the exact
neutrality counterfactual and cannot be selected. B01-B03 differ only in risk budgets. The selected
regime must pass 2x costs, one-session signal delay, factor and sleeve caps tightened/loosened 20%,
rebalance 3/10, and removal of each asset sleeve. Test remains locked until selection and all kill
tests pass.

# v9 Adaptive Orthogonal Alpha Report

## Decision

**REJECTED.** None of K01-K08 passed the frozen train and development profitability gates.
The reused 2025-2026 audit was not loaded, and no orders are allowed. This is a valid negative
research result: the adaptive state model produced positive gross predictive value in the
development interval, but its edge did not cover turnover and borrow. The confidence filter
then became too sparse to form a 50-name strictly neutral portfolio.

The protocol, implementation, and test suite were committed before this evaluation. The source
universe remains `research_snapshot_only`, so no historical PIT or survivor-bias claim is made.

## Frozen candidate result

| Candidate | Definition | Train gross / net Sharpe | Development gross / net Sharpe | Development net CAGR | Decision |
|---|---|---:|---:|---:|---|
| K00 | Legacy fixed Q/R Kalman reference | 0.285 / -0.355 | 0.731 / -0.345 | -0.40% | Reference only |
| K01 | Dynamic R, fixed Q | 0.257 / -0.440 | **0.753** / -0.403 | -0.44% | Reject |
| K02 | Dynamic Q/R | 0.208 / -0.471 | 0.731 / -0.426 | -0.46% | Reject |
| K03 | Four-family orthogonal equal blend | 0.160 / -0.735 | 0.265 / -0.942 | -1.02% | Reject |
| K04 | Four-family train-ICIR blend | 0.271 / -0.434 | 0.544 / -0.591 | -0.64% | Reject |
| K05-K08 | Cost-filtered variants | cash | cash | 0.00% | Reject: too sparse |

K01 is used only as the diagnostic replay because it had the highest robust Sharpe among the
nonempty v9 candidates. It is not a selected or tradeable strategy.

## What the three changes achieved

### 1. Adaptive Kalman Q/R

The implementation is causal: the innovation uses the prior beta, and the Q/R values used on a
session are recorded before that session's return updates the state. A perturbation test verifies
that a large return changes the next session's R but cannot rewrite the current prior Q/R.

Dynamic R varied substantially in development: median prior R was `1.88e-4`, with P99
`1.17e-3`. In the Q/R variant, median Q rose to `2.76e-4`, P99 was `2.09e-3`, and the frozen cap
was reached in the tail. The filter therefore became responsive. That responsiveness did not
create incremental economic value: K02's development gross Sharpe was 0.731, below K01's 0.753
and essentially equal to the legacy reference's 0.731.

### 2. Orthogonal multi-alpha fusion

All four train 3-session mean rank ICs were positive. Frozen ICIR weights were:

| Family | Mean train rank IC | ICIR | Weight |
|---|---:|---:|---:|
| Adaptive Kalman | 0.00762 | 0.05356 | 39.41% |
| Residual reversal | 0.00485 | 0.03821 | 28.12% |
| Volume/price dislocation | 0.00083 | 0.00743 | 5.46% |
| Volatility-conditioned reversal | 0.00308 | 0.03670 | 27.01% |

Daily Gram-Schmidt removed same-session linear exposure to earlier families, but orthogonality
did not guarantee useful portfolio alpha. The ICIR blend's development gross Sharpe was 0.544,
and its net Sharpe was -0.591. The weak volume/price family and unstable state dependence diluted
the Kalman sleeve rather than diversifying it.

### 3. Cost-aligned confidence filter

The four-family train-only calibration slope was `0.000607` expected 3-session return per score
unit. The two-family slope was `0.000352`; neither sign was flipped. Average observation pass
rates were only 2.56% at 1.0x cost, 0.60% at 1.5x, 0.11% at 2.0x, and 0.53% for the two-family
filter. Average development eligible-name counts ranged from 0.29 to 6.51, well below the frozen
50-name construction minimum and 100-name acceptance gate. These candidates correctly held cash.

The filter exposed the central economic problem: the forecast edge is small relative to round-trip
friction. Relaxing the threshold after seeing this result would violate the protocol and would not
be reported as an improvement.

## Cost and portfolio diagnosis

K01 development gross CAGR was 0.80%. Transaction cost consumed 1.47% of initial NAV across the
two-year interval, and borrow consumed another 0.99%. Annual turnover remained about 21x NAV.
Strict neutralization worked: the largest yearly mean pre-neutralization sector exposure was 2.61%
and beta exposure 9.09%; post-neutralization maxima stayed below `3.1e-16`, far inside `1e-8`.

The residual-volatility result contradicts the earlier high-volatility thesis for this alpha:
Q5 contributed -0.12% annualized gross in train and -0.18% in development, while Q1/Q2 were
positive. Market stress was unstable: deep drawdown returned +1.42 bps/day net in train but
-2.96 bps/day in development. Those sign reversals rule out promoting a high-residual-vol or
drawdown overlay from this reused evidence.

## Review figures

- [Alpha IC and rank IC by year](01_alpha_ic_rank_ic_by_year.png)
- [Gross versus net Sharpe](02_gross_vs_net_sharpe.png)
- [Turnover by year](03_turnover_by_year.png)
- [Average holding period](04_average_holding_period.png)
- [Long and short leg PnL](05_long_short_leg_pnl.png)
- [Sector neutralization before and after](06_sector_neutral_before_after.png)
- [Beta neutralization before and after](07_beta_neutral_before_after.png)
- [Cost drag](08_cost_drag.png)
- [Residual-volatility buckets](09_residual_volatility_buckets.png)
- [Cross-sectional dispersion regimes](10_dispersion_regime.png)
- [Market drawdown regimes](11_drawdown_regime.png)
- [Adaptive Q/R distributions](12_adaptive_qr_distributions.png)
- [Cost-confidence pass rate](13_confidence_filter_pass_rate.png)

## Research conclusion

The engineering mechanisms are implemented and behave as specified, but this alpha family is not
strong enough after friction. The next valid step is a new independent alpha source or materially
better data, not more threshold tuning on 2018-2024. Earnings/revision data, intraday
close-to-open/open-to-close decomposition, and historical name-level borrow are the highest-value
additions. Any next historical iteration must label 2023-2026 as reused evidence and earn promotion
only through a prospective shadow period beginning after its specification is frozen.


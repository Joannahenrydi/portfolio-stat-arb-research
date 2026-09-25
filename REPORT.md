# v12 ETF Cross-Asset Relative Value

## Decision

**REJECTED.** The locked 2021-2024 test was not evaluated. No train/validation candidate passed
the frozen profitability gates, and orders remain disabled.

## What worked

The new information source is materially stronger than the equity residual family. Train-only
five-session rank IC was 0.0437 for cross-asset trend and 0.0039 for adaptive Kalman relative
value. Both calibration slopes were positive. Ratio mean reversion had a negative slope and was
blocked without flipping its sign.

All completed portfolios satisfied dollar, equity, duration, credit, commodity and USD neutrality;
the largest numerical exposure residual was below `1.8e-13`.

## What failed

| Candidate | Train net Sharpe | Validation net Sharpe | Validation CAGR | Decision |
|---|---:|---:|---:|---|
| M00 trend | -0.444 | -0.715 | -0.74% | reject |
| M01 Kalman RV | -0.209 | -0.141 | -0.02% | reject |
| M03 equal blend | -0.459 | -0.646 | -0.51% | reject |
| M04 ICIR blend | -0.407 | -0.528 | -0.56% | reject |
| M05 slower ICIR blend | -0.311 | -0.455 | -0.48% | reject |

The optimizer's exact zero constraints removed most of the economically intended trend exposure.
M00 validation gross Sharpe was already negative after projection. The result does not show that
cross-asset trend lacks information; it shows that forcing every macro risk coordinate to zero is
incompatible with a trend sleeve. The next version must use explicit risk budgets and caps rather
than exact neutrality for every factor.

Data are labeled `research_third_party_adjusted`. Yahoo-adjusted SPY returns had 0.995 correlation
with the overlapping Alpaca adjusted series, but this does not replace an Alpaca/SIP replication.


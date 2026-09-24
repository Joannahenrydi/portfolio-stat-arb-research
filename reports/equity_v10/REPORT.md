# v10 Expanded Tail-Selective Adaptive Kalman

## Decision

**REJECTED.** Expanding the source universe from the 341-name complete-history cohort to the
503-name current snapshot materially increased research breadth: 436 names were eligible at
least once in train and 409 in development. Tail concentration and slower rebalancing improved
gross stability and reduced turnover, but no candidate covered modeled trading and borrow costs.
The 2025-2026 reused audit was not loaded and no orders are allowed.

## Candidate evidence

| ID | Construction | Train gross / net Sharpe | Development gross / net Sharpe | Development net CAGR | Turnover | Result |
|---|---|---:|---:|---:|---:|---|
| X00 | 341-name v9 reference | 0.257 / -0.440 | 0.753 / -0.403 | -0.44% | 20.96x | Reference |
| X02 | Expanded, 25% tails, 3-day | -0.116 / -0.916 | **1.266** / -0.213 | -0.15% | 20.96x | Reject |
| X03 | Expanded, 15% tails, 3-day | -0.066 / -0.917 | 1.022 / -0.356 | -0.25% | 20.96x | Reject |
| X04 | Expanded, 25% tails, 5-day | **0.740 / 0.134** | 0.725 / -0.350 | -0.24% | 12.55x | Reject |
| X05 | Expanded, 15% tails, 5-day | 0.637 / 0.066 | 0.574 / -0.415 | -0.28% | 12.55x | Reject |

X01 and the 10-day/EWMA variants fail closed with `MISSING_HELD_RETURN`. The only event is FISV
on 2023-06-07: the local symbol series stops after 2023-06-06 and later resumes in 2025. Without
a verified permanent-ID stitch or terminal return, assuming a return or seamless holding would
fabricate PnL. This data defect is reported rather than silently filled.

## Best structural improvement

X04 is the strongest stable gross result. Relative to X00, its train gross Sharpe rose from 0.257
to 0.740 and annual turnover fell from roughly 21x to 10.7x in train and 12.55x in development.
Development gross Sharpe remained 0.725. The change therefore improved construction quality.

Its economic edge is still too small:

- train gross CAGR 0.74%, transaction drag about 0.38% per year, borrow about 0.23% per year,
  leaving 0.13% net CAGR;
- development gross CAGR 0.48%, transaction drag about 0.44% per year, borrow about 0.27% per
  year, leaving -0.24% net CAGR;
- frozen acceptance requires development net CAGR above 5%, train Sharpe above 0.70, and
  development Sharpe above 0.50.

Increasing gross leverage cannot fix negative development net edge because costs and borrow also
scale. Lowering the assumed costs after seeing the result would not be a valid improvement.

## Interpretation

Expansion improved the opportunity set, and tail selection exposed nonlinear development alpha.
The three-day versions nevertheless have negative train gross Sharpe, while the more stable
five-day version earns less than one percent gross per year. The remaining limitation is alpha
strength, not neutralization or sample count.

With only Alpaca OHLCV, further variants of this same close-to-close residual signal should not be
promoted. A credible next acceptance attempt needs independent information such as PIT earnings
surprises/revisions, a verified intraday dataset for open/close decomposition, and historical
name-level borrow. These can raise forecast edge or remove incorrect uniform borrow assumptions;
more universe/rank/rebalance tuning cannot demonstrate that.

## Figures

- [Net Sharpe gates](01_net_sharpe_gates.png)
- [Development gross versus net Sharpe](02_development_gross_net_sharpe.png)
- [Turnover](03_turnover.png)
- [Executable breadth](04_eligible_breadth.png)


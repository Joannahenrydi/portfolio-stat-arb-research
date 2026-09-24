# v8 - Correlation-Neutral Strategy-Sleeve Allocation

Frozen on 2026-09-24 before any v8 allocation result was computed.

## Research question and source method

Rodrigues and Carrano (2023) treat automated trading strategies as portfolio assets. Their
objective minimizes the absolute Pearson correlation between portfolio log returns and a
market index. Strategy units are nonnegative, and an optional in-sample constraint requires
the portfolio mean return to be at least the index mean return. They refit on walk-forward
windows of three or six months and apply the fitted weights to the following out-sample
window.

v8 tests that idea as an allocation overlay across existing market-neutral strategy sleeves.
It does not reinterpret the paper as a new stock alpha. The paper used Brazilian futures,
30 proprietary ATS streams, integer contract approximations, and did not model this repo's
equity transaction, borrow, liquidity, beta, or sector constraints. Its reported performance
is not a benchmark for this implementation.

## Evidence labels

- 2018-01-02 through 2022-12-30: development train / rolling artifact fit.
- 2023-01-03 through 2024-12-31: development holdout / model selection.
- 2025-01-02 through 2026-09-18: reused audit, descriptive only and inaccessible until
  selection and kill-test locks exist.
- 2026-09-24 onward: pristine prospective shadow.

All historical intervals have influenced earlier research. A survivor may earn only
`DEVELOPMENT_ACCEPTED_PENDING_PROSPECTIVE`; historical results cannot authorize orders.

## Frozen sleeve set

The five ATS analogues are existing, previously defined daily net-return streams:

1. legacy residual reversal;
2. Kalman innovation;
3. volume/volatility dislocation;
4. the legacy multi-alpha blend;
5. v7 R01 short-horizon residual reversal.

The legacy files provide train and development-holdout net returns after their own modeled
costs. v7 R01 is also net of modeled transaction and borrow costs. The allocator never
substitutes gross returns. Each sleeve is an existing strategy definition; v8 does not refit
its alpha parameters.

The source cohort remains `research_snapshot_only`, with current-membership, complete-history
and current-sector bias. Strategy-sleeve allocation cannot repair those data limitations.

## Walk-forward mechanics

All candidates share a common first eligible allocation session after 126 completed market
sessions. Before that session the overlay is cash and those warm-up rows are excluded from
performance statistics. A fit ending at close `t` may first affect the allocation on `t+1`.

During each holding window, sleeve notionals drift with realized sleeve returns. There is no
implicit daily reset to target weights. At a scheduled refit, allocation turnover is the L1
distance between the drifted and new weights. It is capped at 0.60 and charged a conservative
5 bps per unit of allocation turnover, on top of costs already present inside sleeve net
returns. Cross-sleeve stock-level netting is not credited.

For fit-window sleeve return matrix `X`, benchmark return vector `m`, and allocation `a`, the
paper objective is:

```text
minimize |Corr(Xa, m)|
```

Every optimized target satisfies:

```text
sum(a) = 1
0 <= a[i] <= 0.60
sum(abs(a - drifted_previous)) <= 0.60
```

The 60% cap is a predeclared diversification constraint. A candidate is invalid if the
required return floor is infeasible or the numerical solution violates any constraint.

The adapted risk terms are normalized by the contemporaneous equal-weight sleeve portfolio:

```text
variance_ratio = Var(Xa) / Var(Xa_equal)
downside_ratio = Mean(min(Xa, 0)^2) / Mean(min(Xa_equal, 0)^2)
shrinkage      = sum((a - a_equal)^2)
```

No coefficient below may be changed after results are read.

## Frozen experiment matrix

| ID | Fit / hold | Objective | In-sample mean-return floor | Selectable |
|---|---|---|---|---|
| S00 | 126 / 63 | equal weight | none | reference only |
| S01 | 126 / 126 | paper absolute correlation | SPY mean | yes |
| S02 | 63 / 63 | paper absolute correlation | SPY mean | yes |
| S03 | 126 / 63 | paper absolute correlation | SPY mean | yes |
| S04 | 126 / 63 | absolute correlation | zero | yes |
| S05 | 126 / 63 | correlation + 0.25 variance ratio + 0.10 shrinkage | zero | yes |
| S06 | 126 / 63 | correlation + 0.25 downside ratio + 0.10 shrinkage | zero | yes |
| S07 | 126 / 63 | correlation + 0.25 variance ratio + 0.10 shrinkage | equal-weight mean | yes |

The exact paper return floor is retained in S01-S03 even if it proves inappropriate for a
market-neutral US equity portfolio. S04-S07 answer whether the correlation objective adds
value when the return constraint is feasible. Floors are computed on the fit window only.

Optimization uses SLSQP from five deterministic initial points: equal weight, the drifted
previous allocation, and three fixed cyclic 60/10/10/10/10 allocations. The feasible result
with the lowest objective is used. Failure across all starts invalidates the candidate at
that refit; the engine cannot silently keep stale weights or relax a constraint.

## Selection and kill rules

S00 never competes. A candidate must have:

- train net Sharpe greater than 0.70;
- development-holdout net Sharpe greater than 0.50;
- development-holdout net CAGR greater than 5%;
- maximum drawdown no worse than -15% in both intervals;
- annual allocation turnover no greater than 4x;
- maximum absolute realized correlation with SPY no greater than 0.20 in both intervals;
- zero infeasible refits and zero constraint violations.

Rank survivors by the lower of train/development Sharpe, then the worse calendar-year return,
then development CAGR, lower turnover, and ID. The selected candidate must retain positive
development Sharpe under 2x allocation costs, a one-session implementation delay, removal of
each sleeve one at a time, and fit-window shifts of plus/minus 10 sessions. Every kill test
must report its change from S00 in return, volatility, correlation, drawdown and turnover.
Only after both locks are durable may reused-audit data be loaded.

Mandatory reporting includes rolling and full-interval market correlation, allocation weights,
return-floor feasibility, objective components, turnover/cost drag, yearly performance,
concentration, and comparison with both equal weight and the best single sleeve.

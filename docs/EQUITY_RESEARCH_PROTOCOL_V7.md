# v7 — Residual Short-Horizon Alpha Discovery

Frozen on 2026-09-24 before any v7 candidate was evaluated.

> The objective is not to rescue the prior strategy, but to discover a new alpha family
> with independent predictive content.

## Evidence labels and permitted claims

- 2018-01-02 through 2022-12-30 is the **train / artifact-fit** interval.
- 2023-01-03 through 2024-12-31 is the **development holdout / model-selection** interval.
- 2025-01-02 through 2026-09-18 is a **reused audit / descriptive-only** interval.
- 2026-09-24 onward is the **pristine prospective shadow** interval.

Earlier diagnostics exposed information from every historical interval, including the
development holdout and reused audit. Therefore neither interval is pristine OOS. A v7
candidate can earn only `DEVELOPMENT_ACCEPTED_PENDING_PROSPECTIVE`; historical evidence
cannot authorize paper orders or support a claim of validated live alpha. The reused audit
may be read only after both the selection lock and kill-test lock are written.

The price cohort is the v5 341-name, current-snapshot research cohort after the frozen
train-only stale-volume gate. It is not a point-in-time membership dataset and remains
subject to survivor, complete-history and current-sector bias. Every report must retain
the label `research_snapshot_only`.

## Causal residual and signal

At each completed close `t`, estimate each stock's market beta and sector beta from the
previous 126 sessions. The sector factor is the contemporaneous leave-one-out mean of
market-residual returns among eligible sector peers. The residual is:

```text
e[i,t] = r[i,t] - prior_market_component[i,t]
                   - prior_sector_beta[i,t] * sector_factor[i,t]
```

Rolling means are estimated strictly before `t`. A target formed after close `t` first
earns return on session `t+1`; same-session returns never score the portfolio that earned
them.

Cross-sectional z-scores are computed separately on each session and clipped to [-3, 3].
The core forecast is frozen as:

```text
alpha_core[t] = -(0.50 * z(e[t])
                  + 0.30 * z(sum(e[t-2:t]))
                  + 0.20 * z(sum(e[t-4:t])))
```

No momentum, Kalman, or v5 ensemble signal enters v7.

## Eligibility, construction, and costs

Tradable names at a decision close require finite raw close and volume, at least 60 prior
observations, prior 60-session median dollar volume of at least USD 20 million, and a
rank within the top 250 names by that lagged liquidity measure. Membership is recomputed
causally. An ineligible held name must be liquidated through the modeled target; it cannot
vanish from the book or PnL ledger.

Earnings-window exclusion is
`BLOCKED_MISSING_PIT_EARNINGS_CALENDAR`. No current earnings calendar may be projected
backward and missing data must not be interpreted as a non-earnings day.

Every executed target must satisfy dollar, rolling-market-beta and every-sector exposure
within `1e-8`. Failure to construct an executable neutral target invalidates the run.
Shared limits are 1.0 gross, 1% per name, 25% turnover per rebalance and 10 bps of ADV.
Desired targets use the v5 95% execution buffer. Costs are 0.5 bp commission, 2 bp
half-spread, 1 bp slippage, square-root impact coefficient 0.10 and 3% annual common short
borrow. Historical name-level borrow remains unavailable and must not be imputed as an
observed series.

Residual volatility is the prior 60-session residual-return standard deviation and `p`
is its daily cross-sectional percentile. Frozen multipliers are:

- mild: `0.75 + 0.50p`;
- strong: `0.50 + 1.00p`.

The stress scaler uses prior 20-session annualized SPY volatility and its train empirical
CDF percentile `q`. It equals 1.00 through `q=0.80`, declines linearly to 0.50 at `q=0.95`,
declines linearly to 0.25 at `q=0.99`, and remains 0.25 above it. A missing required regime
input makes the run invalid; missing data never means zero exposure.

## Frozen experiment matrix

| ID | Definition | Selectable |
|---|---|---|
| R00 | Simple negative 3-session residual sum; 3-session rebalance | reference only |
| R01 | Core 1/3/5 residual blend; 3-session rebalance | yes |
| R02 | R01 with 2-session rebalance | yes |
| R03 | R01 with 5-session rebalance | yes |
| R04 | R01 + mild residual-volatility multiplier | yes |
| R05 | R01 + strong residual-volatility multiplier | yes |
| R06 | R01 + frozen market-volatility stress scaler | yes |
| R07 | R01 + separate within-side magnitude ranks before neutral projection | yes |
| R08 | R01 + mild residual-volatility multiplier + stress scaler | yes |

R07 ranks positive and negative core forecasts independently by absolute magnitude, maps
each side to [0.5, 1.0], restores its sign, and then applies the same neutral projection.
It does not change the dollar-neutral constraint or reduce aggregate short notional.

## Selection, profitability gate, and kill tests

R00 never competes. A candidate must meet all of the following after modeled costs:

- train Sharpe greater than 0.70;
- development-holdout Sharpe greater than 0.50;
- development-holdout net CAGR greater than 5%;
- train and development-holdout maximum drawdown no worse than -15%;
- annual turnover no greater than 25x NAV;
- zero target neutrality failures.

Rank survivors by the lower of train and development-holdout Sharpe, then the worse full
calendar-year return over 2018–2024, then development-holdout CAGR, then lower turnover,
then experiment ID. This ordering seeks survival across intervals rather than maximum
headline Sharpe.

Before reading the reused audit, the selected candidate must keep positive Sharpe under
2x costs and positive total return under each of: a one-session signal delay, removal of a
deterministic 20% of names, removal of the top 5% development PnL contributors, and the two
adjacent holding/rebalance intervals among 2, 3 and 5 sessions. Each test must report its
change from R01 in gross alpha, transaction cost, borrow, turnover, drawdown and average
gross exposure. Any failure rejects v7. No failed family is tuned until it passes.

Mandatory diagnostics include rank/Pearson IC at 1, 2, 3, 5, 10 and 21 sessions; separate
long/short IC and PnL; annual turnover and holding episodes; cost drag; residual-volatility
buckets; dispersion and market-drawdown regimes; and actual post-trade exposure checks.


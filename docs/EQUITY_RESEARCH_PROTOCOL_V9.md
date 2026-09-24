# v9 - Adaptive State-Space Orthogonal Alpha

Frozen on 2026-09-25 before any v9 candidate was evaluated.

## Objective and evidence status

v9 tests a new alpha family rather than attempting to rescue v5-v8 portfolio weights. It
combines: adaptive state-space innovations, residual reversal, daily price/volume dislocation,
and volatility-conditioned reversal. Signals are orthogonalized cross-sectionally and a
train-only expected-return calibration determines whether forecast edge exceeds modeled costs.

- Train / artifact fit: 2018-01-02 through 2022-12-30.
- Development holdout / model selection: 2023-01-03 through 2024-12-31.
- Reused audit, descriptive only: 2025-01-02 through 2026-09-18.
- Pristine prospective shadow: 2026-09-25 onward.

Earlier research exposed historical diagnostics throughout 2018-2026. Neither train nor the
development holdout is pristine OOS. A historical survivor can earn only
`DEVELOPMENT_ACCEPTED_PENDING_PROSPECTIVE`. Reused-audit data cannot be loaded until selection
and kill-test locks are durable.

The 341-name current-snapshot cohort remains `research_snapshot_only`; v9 does not repair
survivor, complete-history or current-sector bias. The available Alpaca bars are daily. The
volume/price dislocation below is a high-turnover **daily proxy**, not an intraday or HFT claim.

## Adaptive Kalman model

For stock `i` after close `t`, the state-space model is:

```text
beta[i,t] = beta[i,t-1] + eta[i,t],       Var(eta[i,t]) = Q[i,t]
r[i,t]    = beta[i,t] * r_market[t] + eps[i,t], Var(eps[i,t]) = R[i,t]
```

The innovation is formed using the prior beta before `r[i,t]` updates the state. The alpha is
the negative standardized innovation. `R` is updated after the signal using an EWMA of squared
innovations with half-life 20 sessions and a floor of `1e-8`. The warm-up is 60 observations.

Two causal variants are frozen:

- `R-adaptive`: `Q = 1e-5`; `R` follows the lagged EWMA rule.
- `Q/R-adaptive`:

```text
Q[i,t] = clip(
    1e-4 * R[i,t] / prior_market_variance_60[t]
    * clip(prior_market_vol_20[t] / prior_market_vol_60[t], 0.5, 2.0),
    1e-7,
    1e-2,
)
```

Missing prior variance during warm-up uses `Q=1e-5`; it does not use future data. `R` and `Q`
used at `t` are known before observing `r[i,t]`. State covariance is clipped to `[1e-8, 10]`
for numerical stability. No parameter is learned from 2023 onward.

## Alpha families and causal orthogonalization

All families are formed after close `t` and first earn return on `t+1`:

1. `adaptive_kalman`: negative standardized Q/R-adaptive innovation after removing the
   contemporaneous leave-one-out sector mean.
2. `residual_reversal`: negative three-session sum of rolling market/sector residuals.
3. `volume_price_dislocation`: negative residual times lag-safe volume surprise divided by
   prior 60-session volatility.
4. `volatility_conditioned_reversal`: negative three-session residual sum multiplied by the
   clipped ratio of prior 20-session to prior 60-session residual volatility `[0.5, 2.0]`.

Each raw family is cross-sectionally standardized and clipped to `[-3, 3]`. On each decision
session, Gram-Schmidt residualization uses the fixed order above. Each new family is regressed
on an intercept and already orthogonalized families, then its residual is standardized again.
A day with fewer than 50 jointly eligible names is missing, never backfilled.

Two frozen blends are tested:

- equal blend: equal weight across all four orthogonal families;
- ICIR blend: train-only positive `mean daily 3-session rank IC / IC standard deviation`,
  normalized to sum to one. A nonpositive family gets zero weight. If fewer than two families
  remain, the ICIR candidates are blocked.

Forward labels used to fit ICIR must complete by 2022-12-30.

## Cost-aligned confidence filter

The ICIR blend is calibrated on train only. For each day, the three-session forward return is
cross-sectionally demeaned. A zero-intercept ridge slope with penalty `1e-6` maps blend score to
expected three-session residual return. A nonpositive slope blocks cost-filtered candidates;
the sign may not be flipped after seeing development results.

For each name, the frozen round-trip hurdle is:

```text
base = 2 * (0.5 bp commission + 2 bp half-spread + 1 bp slippage)
impact = 2 * 0.10 * prior_daily_vol
         * sqrt((0.005 * USD 100,000) / prior_ADV_60)
borrow = 3% * 3/365 for negative forecasts, otherwise zero
cost_hurdle = base + impact + borrow
```

The assumed one-way trade is 0.5% NAV. A name is eligible only when:

```text
abs(expected_3d_residual_return) >= multiplier * cost_hurdle
```

Missing volatility, ADV, calibration or cost input makes that name ineligible. Existing
positions that become ineligible must be liquidated through the modeled turnover and cost
ledger. The filter never treats missing inputs as zero cost.

## Shared portfolio rules

The strict-neutral v7 engine is reused with a three-session rebalance interval. Actual targets
must satisfy dollar, prior market-beta and every-sector exposure within `1e-8`. Limits remain
1.0 gross, 1% per name, 25% turnover per rebalance, 10 bps ADV participation, and the 95%
target buffer. Portfolio costs remain 0.5 bp commission, 2 bp half-spread, 1 bp slippage,
square-root impact coefficient 0.10 and 3% annual common borrow.

Daily liquidity eligibility is the prior 60-session median dollar volume of at least USD 20
million and top 250 by that measure. PIT earnings and name-level borrow filters remain blocked
because those historical datasets are unavailable.

## Frozen experiment matrix

| ID | Alpha and confidence rule | Selectable |
|---|---|---|
| K00 | fixed-Q/fixed-R legacy Kalman innovation, no confidence filter | reference only |
| K01 | R-adaptive Kalman innovation only | yes |
| K02 | Q/R-adaptive Kalman innovation only | yes |
| K03 | four-family orthogonal equal blend | yes |
| K04 | four-family orthogonal train-ICIR blend | yes |
| K05 | K04 + cost hurdle 1.0x | yes |
| K06 | K04 + cost hurdle 1.5x | yes |
| K07 | K04 + cost hurdle 2.0x | yes |
| K08 | adaptive Kalman + residual reversal equal orthogonal blend + 1.0x cost hurdle | yes |

K00 uses the existing fixed process variance `1e-5` and measurement variance `1e-4` only as a
historical reference. K01-K08 use the new causal implementations above. No threshold is added
or moved after results are read.

## Selection and kill criteria

K00 never competes. A candidate must have train net Sharpe above 0.70, development net Sharpe
above 0.50, development net CAGR above 5%, drawdown no worse than -15% in both intervals,
annual turnover no greater than 25x, zero strict-neutrality failures, and at least 100 average
eligible names in development. Rank survivors by the lower train/development Sharpe, worse
calendar-year return, development CAGR, lower turnover, then ID.

The selected candidate must retain positive development Sharpe under 2x costs, one-session
signal delay, confidence multiplier +/-20%, removal of a deterministic 20% of names, removal
of the top 5% development PnL contributors, and rebalances of 2 and 5 sessions. Any failure
rejects v9. Only after both locks are written may reused-audit data be loaded.

Mandatory outputs include family IC/ICIR and correlation matrices, adaptive Q/R distributions,
calibration slope, pass rate and realized edge/cost ratio by year, gross/net performance,
turnover, long/short PnL, exposure checks, residual-volatility buckets and market regimes.

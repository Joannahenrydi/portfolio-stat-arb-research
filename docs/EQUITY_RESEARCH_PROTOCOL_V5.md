# v5 — Conditional Market-Neutral Stat Arb

Frozen on 2026-09-24 before any v5 candidate was run. The hypotheses were motivated by
diagnostics that included 2025–2026, so that interval is permanently a **reused audit**.
It cannot validate the choices below. The first uncontaminated evidence starts after this
freeze and must be accumulated prospectively.

## Data and development split

- Universe and prices: the same 356-name `research_snapshot_only` cohort used by v2.
  Results remain subject to current-membership, complete-history and current-sector bias.
- Train: 2018-01-02 through 2022-12-30. Train-only quantities include residual-volatility
  distributions, dispersion terciles, alpha IC/correlation filters and ensemble weights.
- Validation: 2023-01-03 through 2024-12-31. It may rank the frozen candidates but cannot
  change their definitions.
- Reused audit: 2025-01-02 through 2026-09-18, read only after the selection lock is saved.
- Prospective shadow start: 2026-09-24. No historical row may be relabeled prospective.

The baseline signal is frozen as 252-session market/sector residual momentum skipping the
most recent 21 sessions, with a five-session rebalance interval. Signals formed at close
`t` first earn the return from `t` to `t+1`.

## Structural rules shared by E01–E14

Every rebalance target must satisfy, within `1e-8`, dollar net = 0, rolling market beta =
0 and every sector exposure = 0. A target is projected into this neutral subspace before
scaling. The current drifted book is separately projected back to the subspace; the
correction plus movement toward the desired target must satisfy 25% turnover, 1% single
name and 10 bps of ADV. If no strictly neutral target is executable, that candidate run is
invalid. It cannot retain a partially neutral target and claim success.

Default gross is 1.0. Costs remain 0.5 bp commission, 2 bp half spread, 1 bp slippage,
square-root impact coefficient 0.10 and 3% annual short borrow. The common borrow rate is
charged but cannot rank short candidates. Historical name-level borrow and hard-to-borrow
data are unavailable, so no proxy may be presented as observed borrow.

Residual volatility is the prior 60-session standard deviation of residual return. Its
cross-sectional percentile `p` is available at decision time. Weight functions retain the
full eligible universe:

- `RV_mild(p) = 0.75 + 0.50p`.
- `RV_strong(p) = 0.50 + 1.00p`.
- `RV_capped(p) = clip(0.60 + 0.80p, 0.70, 1.30)`.

Asymmetric construction multiplies positive centered ranks by 1.0 and negative centered
ranks by either 0.75 (`ASYM_mild`) or 0.50 (`ASYM_strong`) before neutral projection.
Dollar neutrality remains exact; asymmetry changes stock selection and relative weights,
not permitted net market exposure.

Dispersion uses prior-session cross-sectional residual dispersion and its empirical CDF
percentile `q`, estimated on train only. Drawdown uses prior-session SPY drawdown with fixed states:
normal above -5%, moderate from -5% through -15%, stress below -15%.

- `REG_mild`: dispersion multiplier `0.80 + 0.30q`. Drawdown multiplier is 0.80 at
  drawdown >=-5%, rises linearly to 1.00 at -10%, falls linearly to 0.40 at -15%, and
  remains 0.40 below -15%. Product is clipped to [0.35, 1.00].
- `REG_defensive`: dispersion multiplier `0.70 + 0.30q`. Drawdown multiplier is 0.70 at
  drawdown >=-5%, rises linearly to 0.90 at -10%, falls linearly to 0.25 at -15%, and
  remains 0.25 below -15%. Product is clipped to [0.25, 0.90].

These are continuous gross scalers, not binary market-timing switches.

## Frozen experiment matrix

| ID | Signal/portfolio change | Selectable |
|---|---|---|
| E00 | Unmodified v2 baseline replay | reference only |
| E01 | Baseline signal + strict-neutral execution | yes |
| E02 | E01 + RV_mild | yes |
| E03 | E01 + RV_strong | yes |
| E04 | E01 + RV_capped | yes |
| E05 | E01 + ASYM_mild | yes |
| E06 | E01 + ASYM_strong | yes |
| E07 | E01 + REG_mild | yes |
| E08 | E01 + REG_defensive | yes |
| E09 | E01 + RV_mild + REG_mild | yes |
| E10 | E01 + RV_mild + ASYM_mild | yes |
| E11 | E01 + ASYM_mild + REG_mild | yes |
| E12 | E01 + RV_mild + ASYM_mild + REG_mild | yes |
| E13 | Train-qualified low-correlation equal-risk alpha ensemble + strict neutral | conditional |
| E14 | Ridge ensemble fitted once on train and then frozen + strict neutral | conditional |

E13/E14 alpha families are the baseline momentum, residual reversal, Kalman innovation
and volume/volatility dislocation. A non-baseline family can enter only if train mean
five-session rank IC is positive and its daily IC correlation with already admitted
families is below 0.75. E13 cross-sectionally standardizes each admitted family and uses
inverse train IC-volatility weights. E14 fits ridge coefficients once using labels completed
by the 2022-12-30 train boundary and applies that artifact unchanged thereafter. If fewer
than two families qualify, both ensemble experiments are blocked rather than silently
redefined.

## Mandatory diagnostics, not tuning dimensions

- Long and short five-session IC are reported separately, defining the long side as the
  top score half and short side as the bottom score half.
- Forward rank IC and Pearson IC are reported for 1, 5, 10, 21 and 42 sessions.
- Holding-period episodes, long/short PnL, cost attribution and exposure violations are
  reported by year.
- Name-level borrow-aware ranking is `BLOCKED_MISSING_PIT_BORROW_DATA` until an actual
  historical borrow source is available.

## Selection and kill criteria

E00 never competes. A selectable candidate must have positive train and validation net
Sharpe, positive validation CAGR, validation drawdown no worse than -15%, annual turnover
no greater than 25x NAV, and zero strict-neutrality failures. Rank survivors by the lower
of train/validation Sharpe, then the worse full-calendar-year net return across 2018–2024,
then validation Sharpe, then lower turnover, then experiment ID.

The selected candidate must preserve positive validation total return under: doubled
costs, one-session signal delay, +/-10% conditional-weight slope, one slower rebalance
interval, deterministic removal of 20% of names, and removal of the top 5% validation PnL
contributors. Any failure rejects v5. Only after the selection and kill-test lock is
written may the reused audit be evaluated. Reused-audit performance cannot reverse a
development rejection or authorize orders.

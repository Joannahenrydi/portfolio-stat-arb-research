# v11 - Train-Qualified OHLCV Alpha Discovery

Frozen on 2026-09-25 before any v11 feature qualification or candidate result was computed.

## Evidence status and objective

v11 stops tuning the adaptive-Kalman family. It searches a fixed, economically named OHLCV
library for independent cross-sectional alpha, using only 2018-2022 labels for qualification and
weights. Development remains 2023-2024, but it has been viewed by earlier versions and is therefore
reused evidence rather than pristine OOS. The 2025-2026 audit remains unavailable to selection.

The source is the 503-name current S&P 500 snapshot. It is `research_snapshot_only`. A full-file
identity-integrity rule excludes symbols with at least 20 consecutive missing/zero sessions between
their first and last observed bar in the frozen 2017-2026 local collection. This rule is applied
without inspecting returns or alpha and is recorded as complete-history data-quality bias.

Daily eligibility otherwise follows v10: 252 observed returns, raw close at least USD 5, positive
volume, prior median 60-session dollar volume at least USD 10 million, and daily top-400 ADV.

## Frozen feature library

Every raw signal is sector-peer demeaned, cross-sectionally standardized, and clipped to [-3, 3].
Signals formed at close `t` first earn return after `t`.

| Family | Definition | Qualification label |
|---|---|---:|
| `reversal_1` | negative one-day market/sector residual | 5 sessions |
| `reversal_3` | negative three-day residual sum | 5 sessions |
| `reversal_5` | negative five-day residual sum | 5 sessions |
| `momentum_21_5` | residual sum over t-20 through t-5 | 10 sessions |
| `momentum_63_5` | residual sum over t-62 through t-5 | 10 sessions |
| `momentum_126_21` | residual sum over t-125 through t-21 | 21 sessions |
| `momentum_252_21` | residual sum over t-251 through t-21 | 21 sessions |
| `gap_reversal` | negative adjusted open / prior adjusted close minus one | 5 sessions |
| `intraday_reversal` | negative adjusted close / adjusted open minus one | 5 sessions |
| `close_location` | close location inside adjusted high-low range minus 0.5 | 5 sessions |
| `volume_shock_reversal` | negative one-day residual times absolute lag-safe log-volume surprise | 5 sessions |
| `high_252_momentum` | adjusted close divided by current 252-session high | 21 sessions |

For its frozen horizon, a family qualifies only when train mean daily rank IC is positive, ICIR is
at least 0.02, and at least four of calendar years 2018-2022 have positive mean rank IC. Forward
labels must complete by 2022-12-30. Families are sorted by ICIR and greedily admitted only if their
daily train IC correlation with every earlier family is below 0.75. At most four are admitted.

## Frozen blends and portfolios

All blends use the admitted order. Equal blends use equal weights. ICIR blends use positive train
ICIR normalized to one. If the required family count is unavailable, the candidate is blocked.
Tail selection is the v10 rule. Costs and strict dollar/beta/sector neutrality are unchanged.

| ID | Alpha / construction | Selectable |
|---|---|---|
| A00 | exact v10 X04 replay | reference only |
| A01 | highest-ICIR qualified family, 25% tails, rebalance 10 | yes |
| A02 | top two qualified equal blend, 25% tails, rebalance 10 | conditional |
| A03 | up to four qualified ICIR blend, 25% tails, rebalance 10 | conditional |
| A04 | A03, rebalance 21 | conditional |
| A05 | qualified momentum-only ICIR blend, 25% tails, rebalance 21 | conditional |
| A06 | qualified reversal-only ICIR blend, 25% tails, rebalance 5 | conditional |
| A07 | A03, 2.0 gross, 2% name cap, rebalance 21 | conditional |
| A08 | A03, 15% tails, rebalance 10 | conditional |

A07 retains the 25x annual turnover acceptance cap and -15% drawdown gate. It is not accepted merely
for producing a higher CAGR. Every other portfolio uses 1.0 gross and 1% name cap.

## Acceptance and robustness

The frozen v9/v10 gates remain: train net Sharpe above 0.70, development net Sharpe above 0.50,
development net CAGR above 5%, both drawdowns no worse than -15%, annual turnover no greater than
25x, zero neutrality failures, and at least 100 average eligible names. A selected candidate must
retain positive development Sharpe under 2x costs and positive development return under a
one-session signal delay, removal of deterministic 20% of names, removal of top 5% contributors,
tail +/-20%, and neighboring rebalance frequency. Any failure rejects v11.


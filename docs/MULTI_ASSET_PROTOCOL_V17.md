# v17 - Expanded Cross-Asset ETF Trend Ensemble

Frozen on 2026-09-26 before the expanded panel was downloaded or evaluated. The 2017-2020 period
is development data previously viewed in the smaller universe. The 2021-2024 test remains locked.

## New information set

The universe expands from 19 to 45 liquid U.S.-listed ETFs across broad and sector equities,
international country equities, Treasury duration, credit, municipal and emerging-market debt,
metals, diversified/energy/agricultural commodities, and currencies. Membership and coarse risk
loadings are fixed in `scripts/cross_asset_v17_universe.py`. An ETF becomes eligible only after 252
observed sessions and valid lagged 60-session median dollar volume.

## Frozen research rule

Cross-sectional and time-series 3-1/6-1/12-1 trend are calibrated on 2008-2016 exactly as in v16.
Cross-sectional weights `{0%, 25%, 50%, 75%, 100%}` are compared on training data only. The highest
train Sharpe among candidates meeting Sharpe >0.70, CAGR >5%, drawdown >=-15%, and turnover <=25x
is selected. Only that candidate is evaluated in 2017-2020.

The portfolio uses a 10% name cap, 100% gross cap, 10% annual covariance-volatility cap, 60% dollar
net cap, normalized factor caps of 50% equity, 50% duration, 30% credit, 50% commodity and 40% USD,
and sleeve gross caps of 60% equity, 60% rates, 40% credit, 40% metals, 60% commodity and 30% currency.
Costs and liquidity assumptions are unchanged.

Promotion, kill tests and locked-test gates are unchanged from v16. Passing historical gates still
requires Alpaca/SIP replication and prospective paper evidence before any order is allowed.

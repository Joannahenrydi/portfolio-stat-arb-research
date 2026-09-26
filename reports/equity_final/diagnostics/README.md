# Locked Portfolio Diagnostics

Scope: `momentum_252_skip_21`, a 356-stock `research_snapshot_only` cohort from 2018-01-02 through
2026-09-18. Every chart is rebuilt directly from archived Alpaca SIP inputs by
`scripts/plot_portfolio_diagnostics.py`.

1. [Alpha IC / rank IC by year](01_alpha_ic_rank_ic_by_year.png): daily cross-sectional
   Pearson/Spearman correlation between the signal on day `t` and next-session total return,
   averaged by calendar year.
2. [Gross Sharpe vs net Sharpe](02_gross_vs_net_sharpe.png): annual Sharpe calculated from daily
   gross and net returns.
3. [Turnover by year](03_turnover_by_year.png): annual `sum(abs(trade weight))`.
4. [Average holding period](04_average_holding_period.png): completed episodes with unchanged
   holding sign; full-sample average is 94.6 trading sessions.
5. [Long leg / short leg PnL](05_long_short_leg_pnl.png): annual arithmetic gross-PnL contribution
   from each side.
6. [Sector-neutral before/after](06_sector_neutral_before_after.png):
   `sum(abs(sector weight))` for the raw rank portfolio and post-rebalance target.
7. [Beta-neutral before/after](07_beta_neutral_before_after.png): absolute rolling market-beta
   exposure for the raw rank portfolio and post-rebalance target.
8. [Cost drag](08_cost_drag.png): cumulative NAV drag in basis points from transaction and borrow costs.
9. [Residual volatility bucket](09_residual_volatility_bucket.png): annualized gross-PnL
   contribution by quintile of prior-session 60-day residual volatility.
10. [Cross-sectional dispersion regime](10_cross_sectional_dispersion_regime.png): prior-session
    residual cross-sectional standard deviation; Low/Mid/High thresholds use 2018–2022 train
    terciles only.
11. [Market drawdown regime](11_market_drawdown_regime.png): prior-session SPY drawdown grouped into
    above -5%, -5% to -15%, and at or below -15%.

Underlying values are stored in [diagnostics_summary.json](diagnostics_summary.json).

## Main findings

- IC, rank IC and net Sharpe improve in 2024–2026, but net Sharpe is negative in 2019, 2021, 2022
  and 2023; stability is insufficient.
- Borrow is the largest cost component at roughly 140–152 bps in most full years, materially
  reducing gross Sharpe.
- PnL is concentrated in high residual-volatility stocks: Q1 contributes negatively and Q5 most.
- Moderate market drawdowns perform best; normal regimes are slightly negative and SPY drawdowns
  at or below -15% have net Sharpe of -0.81.
- Post-neutralization portfolios retain sector and beta exposure in 2022–2023. The legacy fast
  backtest moved only partway toward neutral targets under turnover and liquidity constraints, so
  it did not restore strict neutrality at each rebalance. This remains a disclosed reason for the
  `REJECTED` decision.

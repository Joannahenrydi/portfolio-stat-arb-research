# v16 - Train-Selected Cross-Asset Trend Ensemble

Frozen on 2026-09-26 before v16 execution. The 2017-2020 period remains a reused development
audit. The 2021-2024 locked test has not been evaluated.

v13 cross-sectional trend produced the stronger return but missed the drawdown gate. v15
time-series trend met the drawdown gate but missed the Sharpe/CAGR gates. v16 tests whether their
different normalization creates useful diversification without adding a new data source.

The two calibrated expected-return scores are blended at cross-sectional weights
`{0%, 25%, 50%, 75%, 100%}`; the remainder is time-series trend. Every blend uses the frozen v13
B03 risk budgets and cost model. All five portfolios are evaluated through 2016 only. A blend is
train-qualified only with net Sharpe >0.70, net CAGR >5%, maximum drawdown no worse than -15%, and
annual turnover <=25x. Among qualified blends, the highest train Sharpe is selected, with lower
turnover and then the experiment ID as deterministic tie-breakers.

Only the single train-selected blend may be evaluated on the reused 2017-2020 development period.
It must pass all original gates. Kill tests cover 2x costs, one-session delay, blend weights +/-25
percentage points, factor/sleeve caps +/-20%, 3/10-session rebalance, and removal of each sleeve.
The 2021-2024 locked test may be loaded once only after every kill test passes.

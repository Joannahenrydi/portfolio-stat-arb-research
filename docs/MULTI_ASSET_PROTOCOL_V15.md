# v15 - Time-Series Cross-Asset Trend Sleeve

Frozen on 2026-09-26 before v15 execution. The 2017-2020 period is a reused development audit.
The 2021-2024 test remains locked.

v13 established that exact neutrality removed useful macro trend exposure. v14 showed that a SPY
stress scaler did not solve trend whipsaw. v15 therefore tests one economically distinct alpha:
continuous time-series momentum. For each ETF, the score is the equal-weight average of its
volatility-normalized 3-1, 6-1 and 12-1 month log return, clipped to [-3, 3]. It is not
cross-sectionally demeaned. The pooled zero-intercept slope against the absolute next-five-session
ETF return is estimated on 2008-2016 only and must be positive.

T00 is the frozen v13 B03 cross-sectional-trend reference and cannot be selected. T01 is the sole
selectable time-series-trend candidate. Both use the same B03 loose risk budgets, five-session
rebalance, cost model, covariance risk cap, liquidity limits and 100% gross cap.

T01 must satisfy the original train and development gates before kill tests. Kill tests are 2x
costs, one-session signal delay, 3/10-session rebalance, factor/sleeve caps +/-20%, and removal of
every asset sleeve. The locked test can be loaded only after every kill test passes.

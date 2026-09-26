# v18 - Train-Selected Blend and Risk Target on Expanded ETFs

Frozen on 2026-09-26 before v18 execution. No expanded-universe portfolio has been evaluated on
2017-2020 or 2021-2024: v17 stopped because no candidate passed the training drawdown gate.

v18 keeps the 45-ETF panel, alphas, costs, liquidity rules, factor/sleeve caps and rebalance schedule
unchanged. It jointly compares the five frozen cross-sectional blend weights `{0, 25, 50, 75,
100}%` with annual covariance-volatility caps `{8, 9, 10}%`, for 15 pre-specified training
candidates. The train gates and deterministic selection rule are unchanged. Only the highest-Sharpe
train-qualified candidate is evaluated on 2017-2020.

If the development gates pass, kill tests include 2x costs, one-session delay, blend +/-25 percentage
points, volatility cap +/-1 percentage point, factor/sleeve caps +/-20%, 3/10-session rebalance,
and removal of every sleeve. The 2021-2024 test remains locked until all kill tests pass.

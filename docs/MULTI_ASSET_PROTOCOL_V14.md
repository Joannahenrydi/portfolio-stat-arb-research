# v14 - Frozen Cross-Asset Trend With Ex-Ante Stress Scaling

Frozen on 2026-09-26 before v14 execution. The 2017-2020 interval has already been viewed in v13
and is now a reused development audit, not a pristine validation set. The 2021-2024 test remains
locked.

## Fixed strategy

v14 freezes the v13 B03 portfolio: v12 cross-sectional ETF trend alpha, five-session rebalance,
loose factor/sleeve risk budgets, 100% gross cap, 15% name cap, 10% annual forecast-volatility cap,
the same liquidity limits, and the same transaction and borrow costs. No alpha or risk-budget
parameter is re-estimated from 2017-2020.

## Stress scaler

The only new candidate uses information available before the rebalance:

- prior-session SPY drawdown from its trailing 252-session high;
- prior-session 20-session annualized SPY realized volatility;
- volatility thresholds fixed at the 90th and 97.5th percentiles of 2008-2016 only.

The exposure scaler is 1.00 normally, 0.75 when drawdown is at most -10% or volatility exceeds the
training 90th percentile, and 0.50 when drawdown is at most -20% or volatility exceeds the training
97.5th percentile. It scales gross, name, dollar-net, factor, sleeve and forecast-volatility caps.
It never increases exposure above the v13 limit.

When a lower risk cap makes the current portfolio infeasible, the rebalance may exceed the ordinary
25% turnover cap only by the minimum amount required to restore the new cap. These forced de-risking
trades retain the full cost model and remain subject to liquidity participation limits.

R00 is the unscaled v13 B03 reference. R01 is the single selectable stress-scaled candidate. R01
must first meet all original acceptance gates on 2008-2016, then on the reused 2017-2020 development
audit. If it passes, kill tests cover 2x costs, one-session signal delay, thresholds shifted by
approximately 20%, three/ten-session rebalance, and removal of every asset sleeve. Only then may
the frozen 2021-2024 test be evaluated once.

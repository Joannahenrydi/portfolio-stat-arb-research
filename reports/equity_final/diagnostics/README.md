# Locked portfolio diagnostics

对象：`momentum_252_skip_21`，356 股票 `research_snapshot_only` cohort，2018-01-02 至
2026-09-18。所有图由 `scripts/plot_portfolio_diagnostics.py` 直接从已归档 Alpaca SIP 输入重建。

1. [Alpha IC / rank IC by year](01_alpha_ic_rank_ic_by_year.png)：信号日 `t` 与下一交易日股票总收益的
   每日横截面 Pearson/Spearman 相关系数，再按年取均值。
2. [Gross Sharpe vs net Sharpe](02_gross_vs_net_sharpe.png)：按日 gross/net return 计算的年度 Sharpe。
3. [Turnover by year](03_turnover_by_year.png)：每年 `sum(abs(trade weight))`。
4. [Average holding period](04_average_holding_period.png)：按持仓符号不变定义、只统计已经结束的 episode；
   全样本平均 94.6 个交易日。
5. [Long leg / short leg PnL](05_long_short_leg_pnl.png)：每年两条腿的算术 gross PnL contribution。
6. [Sector-neutral before/after](06_sector_neutral_before_after.png)：原始 rank portfolio 与调仓后目标的
   `sum(abs(sector weight))`。
7. [Beta-neutral before/after](07_beta_neutral_before_after.png)：原始 rank portfolio 与调仓后目标的绝对
   rolling market-beta exposure。
8. [Cost drag](08_cost_drag.png)：transaction cost 与 borrow cost 对 NAV 的累计 bps。
9. [Residual volatility bucket](09_residual_volatility_bucket.png)：使用前一日可得的 60 日 residual
   volatility 分成五组，显示各组对 gross PnL 的年化贡献。
10. [Cross-sectional dispersion regime](10_cross_sectional_dispersion_regime.png)：使用前一日 residual
    横截面标准差；Low/Mid/High 阈值仅由 2018–2022 training terciles 决定。
11. [Market drawdown regime](11_market_drawdown_regime.png)：使用前一日 SPY drawdown，分为 >-5%、
    -5% 至 -15%、以及 <=-15%。

底层数字保存在 [diagnostics_summary.json](diagnostics_summary.json)。

## 主要诊断

- 2024–2026 的 IC、rank IC 和净 Sharpe 改善，但 2019、2021–2023 净 Sharpe 为负，稳定性不足。
- Borrow 是主要成本来源，大多数完整年份约 140–152 bps；成本使多个年份的 gross Sharpe 明显下降。
- PnL 明显偏向较高 residual-volatility 股票；Q1 贡献为负，Q5 贡献最高。
- 中等市场回撤期表现最好，正常期略负，SPY drawdown <=15% 的 stress 期净 Sharpe 为 -0.81。
- 2022–2023 的“中性化后”仍残留行业与 beta 暴露。旧快速回测在换手和流动性约束下只向中性目标
  部分移动，因此没有在每次调仓后严格恢复中性。这是策略继续 `REJECTED` 的理由，不应隐藏。

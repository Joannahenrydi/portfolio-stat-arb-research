# 组合策略研究结果

状态：拒绝晋级，批准仓位为现金；没有提交订单。

数据来源：Yahoo ETF 原型。

采集 23 个代码，55,776 条日线，2017-01-03 至 2026-09-18。22 个 ETF 为候选，SPY 为基准。

| 区间 | 扣成本收益 | Sharpe | 最大回撤 | 日均换手 |
|---|---:|---:|---:|---:|
| 验证 2023–2024 | -9.67% | -19.99 | -9.67% | 48.21% |
| 时间留出审计 2025–2026 | -7.75% | -14.73 | -7.76% | 48.27% |

固定组合：残差反转 + Kalman 动态相对价值 + 量价偏离，等权混合。按 ETF 经济组与市场 beta 中性化，考虑持仓漂移、下一收盘执行、冲击/价差/佣金/借券成本。

基础统计门槛通过：False；全部压力测试通过：False。详细分策略、分年度和成本分解见 evaluation.json；每日收益、仓位及按名称归因见 CSV。

## 压力测试

| 场景 | 验证收益 | 审计收益 |
|---|---:|---:|
| double_cost | -18.56% | -15.62% |
| delay_one_more | -9.97% | -8.83% |
| horizon_2 | -9.66% | -7.72% |
| horizon_5 | -9.94% | -7.90% |
| alternate_days | -5.67% | -4.79% |
| remove_top_5pct | -9.52% | -7.50% |
| remove_20pct | -8.94% | -7.06% |

## 未完成的生产条件

- Historical PIT security master and delisting reconciliation unavailable
- Retrospective adjusted prices; raw execution/action reconciliation incomplete
- Borrow availability and observed spreads/fills unavailable
- ETF prototype; 200–500-stock universe not collected
- 2025–2026 already inspected in previous project: not blind OOS

本次为 ETF 原型，不是美股 400 标的 PIT 回测。IEX 成交量不能解释为全市场 ADV，结果不能与全市场数据直接等同比较。成本为预设代理；不能据此宣称可交易 alpha。market cap/历史行业标签缺失，未补造。paper_targets.csv 中 research_weight 仅供观察，approved_weight 均为 0。下一步需要本机配置 Alpaca 凭据，获得历史证券主表及公司行为数据，并完成真实 paper 成交对账和 2–3 个月前瞻观察。

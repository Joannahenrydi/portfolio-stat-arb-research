# 组合策略研究结果

状态：拒绝晋级，批准仓位为现金；没有提交订单。

数据来源：Alpaca IEX ETF 数据交叉检查。

采集 23 个代码，35,322 条日线，2018-11-01 至 2026-09-18。22 个 ETF 为候选，SPY 为基准。

| 区间 | 扣成本收益 | Sharpe | 最大回撤 | 日均换手 |
|---|---:|---:|---:|---:|
| 验证 2023–2024 | -2.08% | -5.51 | -2.10% | 8.48% |
| 时间留出审计 2025–2026 | 1.85% | 0.36 | -1.12% | 9.59% |

固定组合：残差反转 + Kalman 动态相对价值 + 量价偏离，等权混合。按 ETF 经济组与市场 beta 中性化，考虑持仓漂移、下一收盘执行、冲击/价差/佣金/借券成本。

基础统计门槛通过：False；全部压力测试通过：False。详细分策略、分年度和成本分解见 evaluation.json；每日收益、仓位及按名称归因见 CSV。

## 压力测试

| 场景 | 验证收益 | 审计收益 |
|---|---:|---:|
| double_cost | -4.15% | -0.16% |
| delay_one_more | -2.48% | -0.12% |
| horizon_2 | -2.21% | 0.26% |
| horizon_5 | -2.37% | 0.64% |
| alternate_days | -1.32% | -0.75% |
| remove_top_5pct | -0.36% | -2.09% |
| remove_20pct | 0.69% | -1.59% |

## 未完成的生产条件

- Historical PIT security master and delisting reconciliation unavailable
- Retrospective adjusted prices; raw execution/action reconciliation incomplete
- Borrow availability and observed spreads/fills unavailable
- ETF prototype; 200–500-stock universe not collected
- 2025–2026 already inspected in previous project: not blind OOS

本次为 ETF 原型，不是美股 400 标的 PIT 回测。IEX 成交量不能解释为全市场 ADV，结果不能与全市场数据直接等同比较。成本为预设代理；不能据此宣称可交易 alpha。market cap/历史行业标签缺失，未补造。paper_targets.csv 中 research_weight 仅供观察，approved_weight 均为 0。下一步需要本机配置 Alpaca 凭据，获得历史证券主表及公司行为数据，并完成真实 paper 成交对账和 2–3 个月前瞻观察。

# Market-neutral US equity research protocol v2

冻结日期：2026-09-23，适用于下一轮新增数据和 prospective 研究。当前已公布的历史结果仍按
`docs/EQUITY_RESEARCH_PROTOCOL.md` 等当轮协议解释；本文件不会被倒推为旧结果的预注册证据。
任何门槛变化必须产生新版本，不能在看到下一轮结果后覆盖本协议。

## 目标与拒绝规则

研究对象是 200–500 只流动性良好的美国股票组成的日频 market-neutral portfolio。Kalman 只是一类
动态 residual alpha，不是 pair strategy。策略必须依次通过数据、alpha、组合、成本、OOS、稳健性和
prospective paper gate；任一阶段失败均为 `REJECTED`，批准仓位为零。

## Week 1–2：数据与 PIT universe

- 决策时间只能读取 `available_at <= decision_time` 且 `effective_at <= decision_time` 的记录。
- 使用永久 security ID；ticker 变更不能创建新历史，退市 ID 不能复用。
- 股票需价格 >= $5、至少 252 个历史 session、过去 60 个 session 覆盖率 >=95%，并按过去 60 日
  median dollar volume 选前 400；默认最低 $20m。
- 数据字段包括 raw/adjusted OHLCV、returns、ADV、volatility、sector/industry、可取得时的 market cap、
  corporate actions、delisting 和 stale/missing flags。缺失的持仓收益不能填零。
- Alpaca 当前 asset master 或当前指数成分只能用于 `research_snapshot_only`。没有历史 security master、
  行业变更、退市与首次发布时间证据时，`pit_verified=false`，不得晋级生产研究。

## Week 3–5：Alpha

- 基础 alpha：1–5 日 market/sector residual return 的横截面反转。
- Kalman alpha：用 t 时点更新前的 beta prior 形成 innovation，再做横截面标准化。
- 第三类：volume surprise 与 volatility dislocation。
- 每个 signal 在 t 收盘形成，最早作用于下一个可执行时点。滚动回归和标准化只能用当时及以前的数据。
- expected-return calibration 只用已完成的历史 label，并保存训练 digest；不能用 validation/test 重估。

## Week 6–7：中性化与 portfolio optimization

每个目标组合满足：dollar net 接近 0，market beta 接近 0，每个行业 exposure 接近 0；可选 size、
value、momentum 因子也作为线性约束。默认 gross <=1、单股 <=1%、单次换手 <=25%、订单 <=0.1%
ADV。目标函数为 `alpha'w - lambda*w'Sigma*w - TC(w)`，协方差只用决策时点以前的收益估计。

## Week 8：成本

基础假设：commission 0.5bp、half-spread 2bp、slippage 1bp、3% 年化 borrow；冲击为
`0.10 * sigma * sqrt(order_notional / ADV)`。借券按自然日计提。所有结果同时报告双倍成本压力。

## Week 9：严格 OOS

按时间做 expanding walk-forward，默认每 21 个 session 重估。Train 为 2018–2022，validation 为
2023–2024；2025–2026 只有在从未查看时才可称 test，否则标为 reused audit。禁止随机切分。
报告净 Sharpe/CAGR、IC、IR、turnover、hit rate、drawdown、capacity、exposure、按年/regime/股票 PnL。

## Week 10：Kill tests

固定检查：成本翻倍、信号延迟一根 bar、轻微参数变化、调仓频率改变、确定性改变 20% universe、
去掉收益贡献最高 5% 股票。任何关键测试使 validation 净收益不为正或暴露/数据 gate 失败，策略拒绝。

## Week 11–12：Paper trading

只有数据与历史研究通过后才允许 Alpaca paper orders。逐日记录 theoretical signal、target、order、fill、
slippage、失败订单、borrow 问题和 PnL attribution。至少积累 42 个 session，目标是连续 2–3 个月。
当前历史研究为 `REJECTED`，因此只能生成 shadow targets，不能提交 paper order。

## 固定接受标准

- Train 和 validation 扣成本 Sharpe 均 >=0.5，净收益为正，最大回撤不超过 20%。
- validation mean rank IC >0，至少 252 个可评估 session，结果不能集中于单一年份或少数股票。
- 所有 kill tests 保持净收益为正；组合中性和风险约束无超限。
- 数据必须通过严格 PIT gate，并完成至少 42 个 prospective paper session。
- 任何失败均输出理由、空订单和 `orders_allowed=false`；不为通过门槛而继续调参。

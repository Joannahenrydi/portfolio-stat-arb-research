# 组合级统计套利 v1

已实现：真实数据采集（Yahoo / Alpaca raw+all）、日度 panel、动态可用性/流动性过滤、三类 alpha、市场 beta 与 ETF 分组中性化、持仓漂移、组合级限额、成本/借券代理、滚动历史参数估计、固定时间切分、七项压力测试、每日收益和名称归因，以及批准仓位为零的 shadow paper 文件。

尚未实现：200–500 股票的历史 PIT 证券主表、历史市值/行业/风格因子、完整退市及公司行为现金账、真实 NBBO/借券可得性、收益预测校准与完整协方差优化器、broker paper 成交与对账、2–3 个月前瞻观察。此版本不能标记为全部 12 周计划已完成。

## 可复现命令

在项目根目录运行：

```bash
.venv/bin/python scripts/collect_portfolio_data.py --provider yahoo
# 凭据通过本机环境注入；不要写入源码或提交到 Git
.venv/bin/python scripts/collect_portfolio_data.py --provider alpaca --feed iex --output output/portfolio_v1/alpaca_new_run
.venv/bin/python scripts/evaluate_portfolio.py
.venv/bin/python scripts/evaluate_portfolio.py --data output/portfolio_v1/alpaca_data --output reports/portfolio_2026-09-20/alpaca_iex
.venv/bin/python -m pytest
```

Alpaca 输出目录必须是新目录，防止覆盖审计数据。无 feed 自动降级；SIP 权限未知时明确使用 IEX。请求范围是 2017-01-01 至 2026-09-19（不含），并不代表服务实际返回完整历史。本次 IEX 多数代码从 2020-07-27 才有日线，且低交易量 ETF 有缺失；不能替代完整训练数据。

原始数据保存在 `output/portfolio_v1/`（Git 忽略）。研究报告保存在 `reports/portfolio_2026-09-20/`。配置 `config/portfolio_v1.json`；冻结规则 `research_protocol.md`。无需新装依赖即可使用现有虚拟环境；独立环境可安装 `pip install -e '.[data,dev]'`。

## 方法的边界

这里使用 score / variance 及线性投影，属于启发式组合构造，不宣称已求解完整的 alpha–covariance–TC 优化问题。滚动回归参数逐日仅使用过去数据；窗口和混合权重固定，没有 walk-forward 超参选择。每日收盘 t 的信号在 t+1 收盘代理执行，t+2 首次贡献收益；手续费/价差/冲击及借券费用扣除。压力场景的隔日调仓允许持仓间漂移，报告相应暴露。最强名称删除使用验证集贡献排名，在验证集上仅是事后诊断，在后续审计期才是固定排除名单。

约束无法通过投影后混合满足时，模型用成本化清仓而非宣称必然可成交；这些风险清仓会让策略拒绝晋级。清仓可超出常规换手或流动性预算，故不能当作可执行回测。真实执行需要后续订单级实现。

所有当前历史输出是 retrospective snapshot；当前复权因子不等于 PIT 数据。Yahoo Close 本身有拆股调整，raw_close_proxy 明确只是代理；Alpaca raw 才是原始价。缺失市值与行业历史保持缺失，ETF 分组是配置标签。2025–2026 已被旧项目查看，因此只是时间留出审计，不是未见过的盲测。行情来源差异不用于择优报告。

官方接口参考：
- https://docs.alpaca.markets/us/reference/stockbars
- https://docs.alpaca.markets/us/docs/market-data-faq
- https://alpaca.markets/sdks/python/api_reference/data/stock/requests.html

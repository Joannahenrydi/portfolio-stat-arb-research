# Cross-Asset Statistical Arbitrage & Trend Research Platform

Research platform for cross-asset trend, equity statistical arbitrage, cost-aware portfolio
construction, walk-forward validation and fail-closed paper execution.

> **Current decision: REJECTED.** Cross-asset trend showed economically meaningful development
> performance under risk-budgeted construction, but failed stability gates across regimes. No
> strategy was promoted and paper orders remain disabled.

## Latest evidence

The project began with U.S. equity residual alpha and evolved into a 45-ETF multi-asset platform
covering equities, rates, credit, metals, commodities and currencies. The main research finding is
that portfolio construction is no longer the binding constraint: price-only alpha loses stability
after 2017.

| Frozen experiment | Train net Sharpe | Development net Sharpe | Net CAGR | Max drawdown | Decision |
|---|---:|---:|---:|---:|---|
| v13 exact-neutral trend | -0.444 | -0.715 | -0.74% development | -4.28% development | Reject |
| v13 risk-budgeted trend | 0.726 | **0.638** | **5.16% development** | -16.43% development | Reject: drawdown |
| v15 time-series trend | 0.688 | 0.547 | 4.33% development | **-14.68% development** | Reject: return gates |
| v16 50/50 trend ensemble | **0.865** | 0.431 | 8.47% train / 3.35% development | -14.25% train / -15.16% development | Reject: instability |
| v18 expanded-universe winner | **0.823** | 0.212 | 7.25% train / 1.41% development | **-13.64% train** / -12.93% development | Reject: instability |

v13 isolated a construction error in v12: forcing equity, duration, credit, commodity and USD
exposure to exactly zero removed the macro trend the strategy was meant to earn. Replacing exact
neutrality with explicit risk budgets raised development net Sharpe from -0.715 to 0.638. Later
versions added time-series momentum, train-only ensemble selection, 45-ETF universe expansion and
risk-target selection. These improved training performance and drawdown control, but the first
development evaluation remained too weak for promotion.

The 2021–2024 locked test was never evaluated in v13–v18 because no candidate passed development
and kill-test gates. The next research cycle requires a distinct information source such as futures
carry/term structure, intraday/overnight decomposition, macro surprises, PIT earnings/revisions or
historical borrow data.

- [v13–v18 consolidated report](reports/cross_asset_v13_v18/REPORT.md)
- [Development NAV and drawdown](reports/cross_asset_v13_v18/development_nav_drawdown.png)
- [Research ladder](reports/cross_asset_v13_v18/research_ladder.png)
- [v18 frozen selection result](reports/cross_asset_v18/development_audit.csv)

## Research capabilities

- Point-in-time-aware data gates and explicit data-quality labels.
- Cross-sectional residual, adaptive Kalman, price/volume and cross-asset trend alpha families.
- Exact equity beta/sector neutrality and risk-budgeted macro portfolio construction.
- Covariance risk, gross/net/name/factor/sleeve, turnover and ADV participation constraints.
- Commission, spread, slippage, square-root impact and borrow-cost accounting.
- Frozen train/development/test protocols, delay/cost/universe kill tests and locked-test controls.
- Alpaca paper execution that emits no orders unless every promotion gate passes.

## Equity research foundation

The original Alpaca equity panel contains SIP raw/all-adjusted daily bars for 503 current candidates
and SPY, a liquid top-400 universe and a 356-name continuous-history research cohort. Those results
remain labeled `research_snapshot_only`: Alpaca's current asset master cannot certify historical
membership or remove survivor bias. The code retains strict bitemporal PIT gates for vendor data
that can provide historical security-master records.

- [策略执行报告](reports/equity_final/REPORT.md)
- [数据审计](reports/equity_v2/data_quality/DATA_STATUS.md)
- [回测结果](reports/equity_v2/backtest/evaluation.json)
- [Kill tests](reports/equity_v2/robustness/kill_tests.csv)
- [晋级决策](reports/equity_final/promotion_decision.json)
- [成本感知优化目标](reports/equity_final/optimized_portfolio/optimized_target.json)
- [组合诊断图](reports/equity_final/diagnostics/README.md)
- [v5 Conditional Market-Neutral 实验报告](reports/equity_v5/REPORT.md)
- [v7 Residual Short-Horizon Alpha Discovery 报告](reports/equity_v7/REPORT.md)
- [v8 论文目标函数策略袖套分配报告](reports/equity_v8/REPORT.md)
- [v9 自适应 Kalman、正交 Alpha 与成本置信度报告](reports/equity_v9/REPORT.md)
- [v10 扩大标的与双尾集中实验报告](reports/equity_v10/REPORT.md)
- [v11 Train-only OHLCV Alpha Discovery 报告](reports/equity_v11/REPORT.md)
- [v12 ETF 跨资产相对价值报告](reports/cross_asset_v12/REPORT.md)
- [v13–v18 跨资产风险预算、趋势融合与扩展标的报告](reports/cross_asset_v13_v18/REPORT.md)

## 目录

- `data/`: 双时点 universe、有效时间/可用时间检查、历史身份门控。
- `features/`: residual reversal、Kalman innovation、volume/volatility alpha 和训练期校准。
- `models/`: 校准与协方差模型的稳定入口。
- `portfolio/`: beta/行业/风格中性化及成本感知组合优化。
- `execution/`: commission、spread、slippage、impact、borrow 成本。
- `risk/`: fail-closed 晋级规则。
- `backtest/`: 持仓漂移、expanding walk-forward、PnL attribution。
- `live/`: 仅 Alpaca paper 的目标与订单计划；被拒绝时返回空订单。
- `reports/`: 数据、回测、稳健性与拒绝记录。
- `tests/`: 防前视、PIT、中性化、成本、400 股票优化和 paper gate 测试。

## 安装与运行

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[alpaca,data,dev]'
pytest -q
ruff check data features models portfolio execution risk backtest live scripts tests
```

数据采集：

```bash
export APCA_API_KEY_ID="..."
export APCA_API_SECRET_KEY="..."
python scripts/collect_week1_alpaca.py \
  --output output/week1-$(date +%Y%m%d) \
  --feed sip --start 2017-01-01 --end 2026-09-19
```

复现已冻结的研究和 shadow 输出：

```bash
python scripts/audit_equity_snapshot.py
python scripts/evaluate_equity_v2.py
python scripts/robustness_equity_v2.py
python scripts/build_optimized_target.py
PYTHONPATH=. python scripts/evaluate_equity_v9.py
PYTHONPATH=. python scripts/diagnose_equity_v9.py
PYTHONPATH=. python scripts/plot_v9_results.py
python -m live.generate_shadow
```

复现冻结的跨资产研究：

```bash
python -m scripts.evaluate_cross_asset_v13
python -m scripts.evaluate_cross_asset_v14
python -m scripts.evaluate_cross_asset_v15
python -m scripts.evaluate_cross_asset_v16
python -m scripts.collect_cross_asset_etfs_v17 \
  --output output/cross_asset_etfs_v17 --start 2007-01-01 --end 2025-01-01
python -m scripts.evaluate_cross_asset_v17
python -m scripts.evaluate_cross_asset_v18
```

连续 2–3 个月的 prospective paper record 必须从未来交易日真实积累，不能用历史回测代替。

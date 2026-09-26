# US Equity Market-Neutral Portfolio Strategy

这是一个**股票组合统计套利平台**。主策略是美股日频横截面 residual mean reversion，
Kalman dynamic residual 和量价/波动错位作为并行 alpha family；系统在组合层统一做净敞口、
市场 beta、行业与可选风格因子中性化，然后进行协方差与成本感知的权重优化。

## 已完成的历史研究

Alpaca SIP raw/all-adjusted 日线覆盖 503 只当前候选和 SPY；流动性筛选 400 只，连续历史研究
cohort 为 356 只。当前最好的冻结候选仍未通过：

| 净结果 | Train 2018–2022 | Validation 2023–2024 | Audit 2025–2026 |
|---|---:|---:|---:|
| CAGR | -1.82% | +0.37% | +6.07% |
| Sharpe | -0.30 | 0.11 | 0.83 |

这些结果是 `research_snapshot_only`：Alpaca 当前 asset master 和当前 ETF 持仓不能证明历史时点
成员资格。代码包含严格的双时点 PIT gate，但只有 Alpaca 时，无法把 2018–2026 历史回测认证为
无幸存者偏差。

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

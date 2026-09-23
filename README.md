# US Equity Market-Neutral Portfolio Research

这是一个**股票组合统计套利平台**。主策略是美股日频横截面 residual mean reversion，
Kalman dynamic residual 和量价/波动错位作为并行 alpha family；系统在组合层统一做净敞口、
市场 beta、行业与可选风格因子中性化，然后进行协方差与成本感知的权重优化。

## Pipeline

```text
Market Data -> PIT Universe -> Features/Alpha -> Neutralization
            -> Portfolio Optimization -> Cost/Risk -> Walk-Forward Backtest
            -> Promotion Gate -> Alpaca Paper Execution -> PnL Attribution
```

组合目标为：

```text
maximize  alpha' w - lambda * w' Sigma w - transaction_cost(w - w_previous)
```

约束包括 gross、net、market beta、sector、single-name、turnover 和 ADV participation。
优化器对 200–500 只股票使用协方差感知的中性子空间解与精确成本 line search，避免逐笔阈值下单。

## 已完成的历史研究

Alpaca SIP raw/all-adjusted 日线覆盖 503 只当前候选和 SPY；流动性筛选 400 只，连续历史研究
cohort 为 356 只。当前最好的冻结候选仍未通过：

| 净结果 | Train 2018–2022 | Validation 2023–2024 | Reused audit 2025–2026 |
|---|---:|---:|---:|
| CAGR | -1.82% | +0.37% | +6.07% |
| Sharpe | -0.30 | 0.11 | 0.83 |

这些结果是 `research_snapshot_only`：Alpaca 当前 asset master 和当前 ETF 持仓不能证明历史时点
成员资格。代码包含严格的双时点 PIT gate，但只有 Alpaca 时，无法把 2018–2026 历史回测认证为
无幸存者偏差。2025–2026 区间也已被反复查看，不再是 pristine blind OOS。

- [12 周执行报告](reports/equity_final/REPORT.md)
- [下一轮冻结研究协议](research_protocol.md)
- [数据审计](reports/equity_v2/data_quality/DATA_STATUS.md)
- [回测结果](reports/equity_v2/backtest/evaluation.json)
- [Kill tests](reports/equity_v2/robustness/kill_tests.csv)
- [晋级决策](reports/equity_final/promotion_decision.json)
- [成本感知优化目标](reports/equity_final/optimized_portfolio/optimized_target.json)

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

Week 1 新数据采集：

```bash
export APCA_API_KEY_ID="..."
export APCA_API_SECRET_KEY="..."
python scripts/collect_week1_alpaca.py \
  --output output/week1-$(date +%Y%m%d) \
  --feed sip --start 2017-01-01 --end 2026-09-19
```

密钥、vendor raw data 和派生价格 panel 不提交 Git。采集器保存 request ID、响应哈希、抓取时间和
明确的 `prospective_only` provenance；它不会用今天的状态伪造过去可获得的信息。

复现已冻结的研究和 shadow 输出：

```bash
python scripts/audit_equity_snapshot.py
python scripts/evaluate_equity_v2.py
python scripts/robustness_equity_v2.py
python scripts/build_optimized_target.py
python -m live.generate_shadow
```

连续 2–3 个月的 prospective paper record 必须从未来交易日真实积累，不能用历史回测代替。

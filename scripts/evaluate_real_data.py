"""Reproduce the prespecified EWA/EWC baseline; no parameter search."""
import json
import platform
from dataclasses import asdict, replace
from importlib.metadata import version
from pathlib import Path

import pandas as pd
from statsmodels.tsa.stattools import adfuller

from pairs_trading.backtest import run_backtest
from pairs_trading.cli import save_result
from pairs_trading.model import StrategyConfig
from pairs_trading.selection import johansen_pair


def main():
    root = Path(__file__).resolve().parents[1]
    out = root / "output/real_api"
    prices = pd.read_csv(out / "market_data/prices.csv", index_col=0, parse_dates=True)
    cfg = StrategyConfig()
    start = "2020-01-01"
    formation = prices.loc[prices.index < start]
    checks = {}
    for symbol in ("EWA", "EWC"):
        level = float(adfuller(formation[symbol], autolag="AIC")[1])
        diff = float(adfuller(formation[symbol].diff().dropna(), autolag="AIC")[1])
        checks[symbol] = {"adf_level_p": level, "adf_difference_p": diff,
                          "passes_i1": level > .05 and diff < .05}
    checks["johansen_95_pass"] = johansen_pair(formation.EWA, formation.EWC)
    checks["eligible_by_original_selection"] = (
        all(checks[s]["passes_i1"] for s in ("EWA", "EWC")) and checks["johansen_95_pass"]
    )
    summaries = {}
    baseline = None
    for name, scenario in (
        ("baseline", cfg),
        ("zero_cost", replace(cfg, commission_bps=0, slippage_bps=0)),
        ("double_cost", replace(cfg, commission_bps=2, slippage_bps=4)),
    ):
        result = run_backtest(prices, "EWA", "EWC", scenario, trade_start=start)
        save_result(result, str(out / name))
        summaries[name] = result.summary
        if name == "baseline":
            baseline = result
    annual = []
    previous = cfg.initial_capital
    for year, frame in baseline.equity.groupby(baseline.equity.index.year):
        last = float(frame.equity.iloc[-1])
        annual.append({"year": int(year), "return": last / previous - 1})
        previous = last
    audit = {"config": asdict(cfg), "test_start": start, "diagnostics": checks,
             "runtime": {"python": platform.python_version(),
                         **{p: version(p) for p in ("numpy", "pandas", "statsmodels", "yfinance")}},
             "scenarios": summaries, "annual_returns": annual}
    (out / "evaluation.json").write_text(json.dumps(audit, indent=2) + "\n")
    lines = [
        "# EWA / EWC 真实 API 数据回测", "",
        "数据：Yahoo Finance，经 yfinance 只读下载；复权日收盘价。",
        f"共同交易日：{prices.index[0].date()} 至 {prices.index[-1].date()}，共 {len(prices)} 行。",
        "2015—2019 年用于初始化、在线预热及测试前筛选诊断；2020—2025 年为固定参数历史留出测试。",
        "首 126 根日线估计初始参数，此后 Kalman 因果在线更新。未根据测试收益调参或更换交易对。",
        "这是历史留出评估，不是已运行六年的前瞻虚拟盘，也未验证新线性预测模型。", "",
        "## 结果", "",
        "| 场景 | 总收益 | 最大回撤 | Sharpe（无风险利率为零） | 开仓次数 |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, summary in summaries.items():
        lines.append(f"| {name} | {summary['total_return']:.2%} | {summary['max_drawdown']:.2%} | "
                     f"{summary['sharpe_zero_rf']:.3f} | {summary['entries']} |")
    lines += ["", "基准每次成交按每腿名义金额收取手续费 1 bp＋滑点 2 bp；double_cost 为两倍。",
              "初始账户 100,000 美元；每对开仓总名义金额上限为当时权益的 20%。两腿等金额、整数股。",
              "收益与回撤均以整个账户计，闲置现金未计息；期末持仓按收盘价标记，未强制清仓。", "",
              "## 年度收益", "", "| 年份 | 账户收益 |", "|---|---:|"]
    lines += [f"| {r['year']} | {r['return']:.2%} |" for r in annual]
    lines += ["", "## 测试前筛选诊断", "",
              "```json", json.dumps(checks, indent=2), "```", "",
              "若 eligible_by_original_selection 为 false，严格执行原筛选规则应不交易；上述强制交易结果仅是示例配对的诊断基线。",
              "筛选只使用 2020 年之前的数据。协整检验通过也不保证未来价差稳定。", "",
              "## 解释与限制", "",
              "- 本次固定参数测试中，零成本情形仍亏损，未发现该基线的盈利证据；不能据此否定所有配对或 Kalman 方法。",
              "- Kalman 用于价差信号；实际等金额仓位并非 Kalman 比例对冲，不应称为严格 Beta 中性。",
              "- 信号在下一根日线收盘执行，双腿假定同时成交；没有盘口、借券可用性、借券费和冲击模型。",
              "- 复权价是总回报研究代理，不是可直接成交的价格；整数股、股息和拆股现金流尚未按真实账本处理。",
              "- 当前复权历史不是 point-in-time 数据，跨市场日期对齐也不能代表同步报价。",
              "- 未筛选全球市场；EWA/EWC 来自本项目已有示例，避免根据本次结果挑选获胜配对。",
              "- 原始字段、对齐数据、下载时间及 SHA256 见 market_data；交易、信号和权益见各场景文件夹。",
              "- 修复了资本不足时仅一腿数量为零却残留另一腿虚假持仓的问题；新增测试区间控制，预热期不计入绩效。", "",
              "## 复现", "", "```bash",
              ".venv/bin/python -m pairs_trading.cli download --symbols EWA EWC --start 2015-01-01 --end 2026-01-01 --output output/real_api/market_data",
              ".venv/bin/python scripts/evaluate_real_data.py", "```", "",
              "接口文档：https://ranaroussi.github.io/yfinance/", ""]
    (out / "REPORT.md").write_text("\n".join(lines))
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()

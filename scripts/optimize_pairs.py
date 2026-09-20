"""Bounded chronological search. Run `select` BEFORE downloading fresh holdout."""
import argparse
import hashlib
import itertools
import json
from dataclasses import asdict, replace
from pathlib import Path

import pandas as pd

from pairs_trading.research import Costs, Parameters, load_market, research_signals, simulate

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/optimization"
COSTS = Costs()
STRESS = replace(COSTS, commission_per_share=.01, minimum_per_order=2,
                 slippage_bps=4, sell_fee_bps=.6, borrow_annual=.06)


def candidates():
    for model in ("kalman", "rolling_ols"):
        for window, q, entry, exit_z, holding, reverting in itertools.product(
            (20, 60, 120), (1e-7, 1e-5, 1e-3) if model == "kalman" else (1e-5,),
            (1., 1.5, 2., 2.5), (.1, .5), (10, 30), (False, True),
        ):
            yield Parameters(model, window, q, entry, exit_z, 4., holding, reverting)


def save_run(name, result):
    summary, curve, trades = result
    path = OUT / name
    path.mkdir(parents=True, exist_ok=True)
    curve.to_csv(path / "equity.csv")
    trades.to_csv(path / "trades.csv", index=False)
    (path / "summary.json").write_text(json.dumps(summary, indent=2))


def select():
    OUT.mkdir(parents=True, exist_ok=True)
    protocol = {
        "pair": ["EWA", "EWC"], "capital": 100000, "maximum_historical_drawdown": .05,
        "training": ["2016-01-01", "2019-12-31"],
        "validation": ["2020-01-01", "2022-12-31"],
        "previously_seen_audit": ["2023-01-01", "2025-12-31"],
        "fresh_holdout": ["2026-01-01", "2026-09-18"],
        "search_candidates": len(list(candidates())), "training_top_k": 12,
        "fractions": [0, .05, .1, .2, .35, .5, .75, 1.],
        "costs": asdict(COSTS), "stress_costs": asdict(STRESS),
        "selection": "train Sharpe top 12 (>=8 trades); rank by minimum train/validation Sharpe; positive return and >=4 validation trades; choose fraction maximizing validation return under base/stress positive returns and <=5% drawdowns in train and validation; cash otherwise",
        "promotion": "fresh holdout positive base and stressed return, >=5 entries and <=5% drawdown in both; previously seen 2023-2025 must also be positive under base/stress and <=5% drawdown; otherwise no positive allocation supported",
        "caveat": "2020-2025 baseline already inspected in prior turn; only 2026 is newly reserved.",
    }
    (OUT / "protocol.json").write_text(json.dumps(protocol, indent=2))
    market = load_market(ROOT / "output/real_api/market_data").loc[:"2022-12-31"]
    cached = {}

    def signals(p):
        key = (p.model, p.window, p.process_variance)
        if key not in cached:
            cached[key] = research_signals(market, p)
        return cached[key]

    rows = []
    for number, p in enumerate(candidates()):
        result = simulate(market, signals(p), p, *protocol["training"])[0]
        rows.append({"id": number, **asdict(p), **{"train_" + k: v for k, v in result.items()}})
    search = pd.DataFrame(rows)
    search.to_csv(OUT / "training_search.csv", index=False)
    finalists = search.loc[(search.train_entries >= 8) & (search.train_drawdown >= -.05)]
    finalists = finalists.sort_values(["train_sharpe", "id"], ascending=[False, True]).head(12)
    evaluated = []
    parameters = list(candidates())
    for _, row in finalists.iterrows():
        p = parameters[int(row.id)]
        result = simulate(market, signals(p), p, *protocol["validation"])[0]
        evaluated.append({**row.to_dict(), **{"validation_" + k: v for k, v in result.items()},
                          "robust_score": min(row.train_sharpe, result["sharpe"])})
    validation = pd.DataFrame(evaluated).sort_values("robust_score", ascending=False)
    validation.to_csv(OUT / "validation_candidates.csv", index=False)
    best = validation.iloc[0]
    p = parameters[int(best.id)]
    eligible = validation.loc[(validation.train_return > 0) & (validation.validation_return > 0)
                              & (validation.validation_entries >= 4)]
    if len(eligible):
        p = parameters[int(eligible.iloc[0].id)]
    amounts = []
    for fraction in protocol["fractions"]:
        row = {"fraction": fraction, "initial_gross_budget": fraction * 100000}
        good = bool(len(eligible))
        for stage in ("training", "validation"):
            for label, costs in (("base", COSTS), ("stress", STRESS)):
                result = simulate(market, signals(p), p, *protocol[stage], fraction=fraction, costs=costs)[0]
                row.update({stage + "_" + label + "_" + k: v for k, v in result.items()})
                good = good and result["return"] > 0 and result["drawdown"] >= -.05
        row["eligible"] = bool(good and fraction > 0)
        amounts.append(row)
    sizes = pd.DataFrame(amounts)
    sizes.to_csv(OUT / "sizing.csv", index=False)
    allowed = sizes.loc[sizes.eligible]
    selected_fraction = float(allowed.sort_values("validation_base_return", ascending=False).iloc[0].fraction) if len(allowed) else 0.0
    selected = {"parameters": asdict(p), "selected_fraction": selected_fraction,
                "selected_initial_gross": selected_fraction * 100000,
                "diagnostic_fraction": selected_fraction or .2,
                "reason": "validation/stress constraints passed" if selected_fraction else "no positive allocation passed validation and stress constraints",
                "protocol_sha256": hashlib.sha256((OUT / "protocol.json").read_bytes()).hexdigest()}
    (OUT / "frozen_selection.json").write_text(json.dumps(selected, indent=2))
    print(json.dumps(selected, indent=2), flush=True)


def evaluate():
    frozen_bytes = (OUT / "frozen_selection.json").read_bytes()
    chosen = json.loads(frozen_bytes)
    protocol = json.loads((OUT / "protocol.json").read_text())
    p = Parameters(**chosen["parameters"])
    market = load_market(OUT / "market_data")
    signals = research_signals(market, p)
    fraction = chosen["diagnostic_fraction"]
    results = {}
    for stage in ("training", "validation", "previously_seen_audit", "fresh_holdout"):
        for label, costs in (("base", COSTS), ("stress", STRESS)):
            name = stage + "_" + label
            result = simulate(market, signals, p, *protocol[stage], fraction=fraction, costs=costs)
            results[name] = result[0]
            save_run(name, result)
    baseline = Parameters(window=60, process_variance=1e-5, entry=1., exit=.15,
                          stop=3.5, holding=60, reverting_only=False, cost_buffer=0)
    comparison = {}
    for label, parameter in (("original_thresholds", baseline), ("selected_candidate", p)):
        sig = research_signals(market, parameter)
        result = simulate(market, sig, parameter, "2020-01-01", "2025-12-31",
                          fraction=.2, costs=COSTS)
        comparison[label] = result[0]
        save_run("comparison_" + label, result)
    promoted = chosen["selected_fraction"] > 0
    for stage in ("previously_seen_audit", "fresh_holdout"):
        for label in ("base", "stress"):
            r = results[stage + "_" + label]
            promoted = promoted and r["return"] > 0 and r["drawdown"] >= -.05
            if stage == "fresh_holdout":
                promoted = promoted and r["entries"] >= 5
    audit = {"frozen_selection_sha256": hashlib.sha256(frozen_bytes).hexdigest(),
             "selection": chosen, "evaluated_fraction": fraction, "results": results,
             "same_engine_comparison_2020_2025": comparison,
             "passes_promotion": bool(promoted),
             "supported_initial_gross_allocation": chosen["selected_initial_gross"] if promoted else 0}
    (OUT / "evaluation.json").write_text(json.dumps(audit, indent=2))
    lines = ["# 费用后策略优化与仓位实验", "",
             f"账户：$100,000；历史回撤约束 5%；搜索 {protocol['search_candidates']} 组参数，训练前 12 名进入验证。",
             "训练：2016—2019；验证：2020—2022；已看过的历史复核：2023—2025；新留出区间：2026-01-01 至 2026-09-18。",
             "此前已查看过 2020—2025 原策略结果，因此该区间不能声称完全未被观察；2026 数据在选择冻结后才下载。", "",
             f"验证阶段选定初始双腿总金额：${chosen['selected_initial_gross']:,.0f}。",
             f"最终满足预设检验的初始双腿总金额：${audit['supported_initial_gross_allocation']:,.0f}。",
             f"下表按账户 {fraction:.0%} 总名义仓位运行；若验证选择为零，这只是候选策略诊断，实际选择是持有现金。", "",
             "| 区间 | 成本 | 收益 | 最大回撤 | Sharpe | 开仓 |",
             "|---|---|---:|---:|---:|---:|"]
    for name, r in results.items():
        lines.append(f"| {name.rsplit('_', 1)[0]} | {name.rsplit('_', 1)[1]} | {r['return']:.2%} | {abs(r['drawdown']):.2%} | {r['sharpe']:.2f} | {r['entries']} |")
    lines += ["", "## 同记账与成本下的参数改进对照", "",
              "2020—2025；均为 20% 仓位、实际价格加分红记账、比例对冲、同一成本模型。原阈值对照也经过配仓和记账修正，因此不是上次旧引擎结果的直接复刻。", "",
              "| 版本 | 收益 | 最大回撤 | 开仓 |", "|---|---:|---:|---:|"]
    for label, r in comparison.items():
        lines.append(f"| {label} | {r['return']:.2%} | {abs(r['drawdown']):.2%} | {r['entries']} |")
    lines += ["", "## 初始总金额敏感性", "",
              "以下为冻结候选在验证期 2020—2022 的结果；每次开仓按当时权益乘仓位比例，金额随权益变化。", "",
              "| 初始双腿总金额 | 默认成本收益 | 双倍成本收益 |", "|---|---:|---:|"]
    for _, row in pd.read_csv(OUT / "sizing.csv").iterrows():
        lines.append(f"| ${row.initial_gross_budget:,.0f} | {row.validation_base_return:.2%} | {row.validation_stress_return:.2%} |")
    lines += ["", "## 冻结参数", "", "```json", json.dumps(asdict(p), indent=2), "```", "",
              "## 成本及配仓", "",
              "每腿每次成交 max($1, 股数×$0.005)，卖出额另加 0.3 bp 费用预留；滑点每次 2 bp；空头年化借券 3%，按日历天计费。",
              "压力测试把上述费用、滑点与借券率全部加倍。佣金参考 IBKR Fixed 的形式，但不是账户精确报价或历史费率复原。",
              "https://www.interactivebrokers.com/en/pricing/commissions-stocks.php",
              "https://www.interactivebrokers.com/en/pricing/short-sale-cost.php", "",
              "信号在 t 日收盘产生，t+1 日收盘成交；对冲比例取信号时已知值。qY=floor(G/(PY+h×PX))，qX=-floor(h×qY)，方向随信号改变。",
              "价格信号使用从初始价格向前累积的总回报指数，避免使用事后复权因子；对冲比例换算成实际股数。",
              "实际收盘价记账，持仓分红单独计入，多头收取、空头支付；期末强制平仓并计费。含拆股数据主动拒绝。",
              "入场要求预计价差偏离金额超过预计往返交易及借券成本的 1.5 倍，并可要求偏离开始收敛。该偏离只是筛选代理，不是保证可赚取的收益。", "",
              "## 解释边界", "",
              "- 最优金额仅指本账户、候选仓位网格、费用与历史风险约束下的条件最优；不是全局最优本金。",
              "- 零金额表示未找到满足门槛的正投入；不能据此证明所有策略无效。",
              "- 5% 是历史筛选上限，不是未来回撤保证或盘中止损承诺。",
              "- 尚未模拟借券可用性、动态借券率、买卖盘口、双腿异步成交、冲击、融资与税费的完整规则。",
              "- 无杠杆开仓，总名义金额不超过账户权益；等风险中性仍需另外验证市场因子暴露。",
              "- 2026 不足一年，即使通过也只支持继续虚拟盘，不构成稳定盈利证明。",
              "- 搜索全部结果保存在 training_search.csv，验证、仓位与各阶段交易记录同目录保存。", ""]
    (OUT / "REPORT.md").write_text("\n".join(lines))
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["select", "evaluate"])
    args = parser.parse_args()
    select() if args.action == "select" else evaluate()

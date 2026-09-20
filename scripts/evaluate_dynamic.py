"""Research dynamic forecasts using chronological development and retrospective audits."""
import itertools
import json
import os
import zipfile
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd

from pairs_trading.dynamic import (
    DynamicParameters,
    desired_position,
    dynamic_signals,
    simulate_dynamic,
)
from pairs_trading.research import Costs, load_market

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/dynamic"
BASE = Costs()
STRESS = replace(BASE, commission_per_share=.01, minimum_per_order=2,
                 slippage_bps=4, sell_fee_bps=.6, borrow_annual=.06)
PERIODS = {"training": ("2017-01-01", "2019-12-31"),
           "validation": ("2020-01-01", "2022-12-31"),
           "audit": ("2023-01-01", "2025-12-31"),
           "research_2026": ("2026-01-01", "2026-09-18")}


def save_result(name, result):
    directory = OUT / name
    directory.mkdir(exist_ok=True)
    summary, curve, trades = result
    (directory / "summary.json").write_text(json.dumps(summary, indent=2))
    curve.to_csv(directory / "equity.csv")
    trades.to_csv(directory / "trades.csv", index=False)


def make_charts(market, signals, dynamic, stressed, evaluation, sizes):
    os.environ.setdefault("MPLCONFIGDIR", str(OUT / ".mpl-cache"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titleweight": "bold", "figure.facecolor": "#f7f9fc",
                         "axes.facecolor": "white", "grid.alpha": .2, "svg.fonttype": "none"})
    blue, teal, red = "#3156a6", "#007e87", "#cc5262"
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("EWA / EWC · Dynamic strategy research", x=.07, ha="left", fontsize=21, fontweight="bold")
    fig.text(.07, .924, "$100,000 account · 20% maximum entry gross exposure · fees, slippage and borrow included", color="#526078")
    for label, result, color in (("Dynamic · doubled costs", stressed, blue), ("Dynamic · base costs", dynamic, teal)):
        curve = result[1]
        axes[0, 0].plot(curve.index, curve.equity/100000-1, label=label, color=color, lw=1.7)
        axes[0, 1].plot(curve.index, curve.equity/curve.equity.cummax()-1, label=label, color=color, lw=1.5)
    axes[0, 0].axhline(0, color="#778397", lw=.8)
    axes[0, 0].set_title("Cumulative account return · 2020–2026")
    axes[0, 1].set_title("Drawdown from account high")
    for ax in axes[0]:
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.legend(loc="lower left", frameon=False)
        ax.xaxis.set_major_locator(mdates.YearLocator(2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.grid(axis="y")
    recent = dynamic[1].loc["2025-01-01":]
    axes[1, 0].plot(recent.index, recent.prediction, color=teal, lw=1.3, label="Predicted pair return")
    axes[1, 0].plot(recent.index, recent.entry_threshold, color=red, lw=1, label="Dynamic entry threshold")
    axes[1, 0].plot(recent.index, -recent.entry_threshold, color=red, lw=1)
    axes[1, 0].axhline(0, color="#778397", lw=.8)
    axes[1, 0].set_title("Forecast and cost / volatility threshold")
    axes[1, 0].yaxis.set_major_formatter(PercentFormatter(1))
    axes[1, 0].legend(frameon=True, facecolor="white", framealpha=.95, loc="lower left")
    recent_signal = signals.loc["2025-01-01":]
    axes[1, 1].plot(recent_signal.index, recent_signal.weight_x, color=blue, lw=1.5, label="EWA share of gross dollars")
    axes[1, 1].plot(recent_signal.index, recent_signal.weight_y, color=teal, lw=1.5, label="EWC share of gross dollars")
    axes[1, 1].set_ylim(.35, .65)
    axes[1, 1].yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    axes[1, 1].set_title("Constrained risk-minimizing dollar split")
    axes[1, 1].legend(frameon=False, loc="lower left")
    for ax in axes[1]:
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.grid(axis="y")
    fig.text(.07, .026, "Retrospective research; no account resets. Both curves use this version. Doubled costs rerun filters and change trade counts.", fontsize=10, color="#526078")
    fig.subplots_adjust(left=.07, right=.97, bottom=.09, top=.86, wspace=.25, hspace=.37)
    for suffix in ("png", "svg"):
        fig.savefig(OUT / f"performance.{suffix}", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    fig.suptitle("Validation, position sizing and annual performance", x=.06, ha="left", fontsize=19, fontweight="bold")
    labels = ["2017–19\nTrain", "2020–22\nValidate", "2023–25\nAudit", "2026 YTD\nResearch"]
    locations = np.arange(4)
    for offset, cost, color in ((-.18, "base", teal), (.18, "stress", red)):
        numbers = [evaluation[stage+"_"+cost]["return"] for stage in PERIODS]
        axes[0].bar(locations+offset, numbers, .36, color=color, label=cost.title())
    axes[0].set_xticks(locations, labels)
    axes[0].set_title("Separate periods · 20% cap")
    axes[0].legend(frameon=False)
    axes[1].plot(sizes.cap*100, sizes.validation_return, "o-", color=teal, label="Base costs")
    axes[1].plot(sizes.cap*100, sizes.validation_stress_return, "o-", color=red, label="Doubled costs")
    axes[1].set_xlabel("Maximum gross allocation (% of account)")
    axes[1].set_title("2020–2022 sizing sensitivity")
    axes[1].legend(frameon=False)
    annual, prior = [], 100000.
    for year, part in dynamic[1].groupby(dynamic[1].index.year):
        last = float(part.equity.iloc[-1])
        annual.append((year, last/prior-1))
        prior = last
    axes[2].bar([str(y) for y, _ in annual], [r for _, r in annual], color=[teal if r >= 0 else red for _, r in annual])
    axes[2].set_title("Dynamic account · annual return")
    axes[2].tick_params(axis="x", rotation=45)
    for ax in axes:
        ax.axhline(0, color="#778397", lw=.8)
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.grid(axis="y")
    fig.text(.06, .045, "Base: $1 minimum commission per leg/order, 2 bp slippage, 3% annual borrow. Stress doubles modeled costs. 2026 is partial-year.", fontsize=9, color="#526078")
    fig.text(.06, .016, "Stress reruns the cost-aware entry filters, so trade counts change. The report also shows doubled costs on the identical original orders.", fontsize=9, color="#526078")
    fig.subplots_adjust(left=.06, right=.98, bottom=.19, top=.8, wspace=.35)
    for suffix in ("png", "svg"):
        fig.savefig(OUT / f"validation_and_sizing.{suffix}", dpi=180)
    plt.close(fig)
    for path in OUT.glob("*.svg"):
        path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines())+"\n")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    market = load_market(ROOT / "output/optimization/market_data")
    candidates = [DynamicParameters(window, horizon, ridge, confidence, features)
                  for window, horizon, ridge, confidence, features in itertools.product(
                      (252, 504), (5, 10), (.1, 1.), (0., .1, .25), ("simple", "full", "kalman"))]
    protocol = {"candidate_count": len(candidates), "periods": PERIODS,
                "capital": 100000, "diagnostic_cap": .2, "max_historical_drawdown": .05,
                "costs": asdict(BASE), "stress_costs": asdict(STRESS),
                "selection": "Top 12 training Sharpe candidates with >=8 entries; select highest min(train,validation) Sharpe among positive train/validation and positive doubled-cost validation, otherwise diagnostic best with zero allocation.",
                "allocation": "0/5/10/20/35/50/75/100% gross cap; maximize validation return if train,validation and stressed validation positive with <=5% historical drawdown; subsequent audit must also pass.",
                "warning": "All dates including 2026 were observed before this research; no untouched test claim.",
                "research_history": "Initial 48 ridge configurations produced weak aggregate improvement and failed validation. Added 24 Kalman-z symmetric reversion configurations. This is iterative exploratory research.",
                "execution": "Prior-close integer share quantities; next-close fills; monthly ridge trained only on matured next-close-entry horizon labels."}
    (OUT / "protocol.json").write_text(json.dumps(protocol, indent=2))
    cached = {}

    def signals(p):
        key = (p.train_window, p.horizon, p.ridge, p.features)
        if key not in cached:
            cached[key] = dynamic_signals(market, p)
        return cached[key][0]

    rows = []
    for number, p in enumerate(candidates):
        r = simulate_dynamic(market, signals(p), p, *PERIODS["training"])[0]
        rows.append({"id": number, **asdict(p), **{"train_"+k: v for k, v in r.items()}})
    training = pd.DataFrame(rows)
    training.to_csv(OUT / "training_search.csv", index=False)
    finalists = training.loc[training.train_entries >= 8].sort_values("train_sharpe", ascending=False).head(12)
    if finalists.empty:
        finalists = training.sort_values("train_sharpe", ascending=False).head(12)
    validation = []
    for _, row in finalists.iterrows():
        p = candidates[int(row.id)]
        base = simulate_dynamic(market, signals(p), p, *PERIODS["validation"])[0]
        stress = simulate_dynamic(market, signals(p), p, *PERIODS["validation"], costs=STRESS)[0]
        validation.append({**row.to_dict(), **{"validation_"+k: v for k, v in base.items()},
                           "validation_stress_return": stress["return"],
                           "robust_score": min(row.train_sharpe, base["sharpe"])})
    validation = pd.DataFrame(validation).sort_values("robust_score", ascending=False)
    validation.to_csv(OUT / "validation_candidates.csv", index=False)
    eligible = validation.loc[(validation.train_return > 0) & (validation.validation_return > 0)
                              & (validation.validation_stress_return > 0) & (validation.validation_entries >= 8)]
    selected_row = eligible.iloc[0] if len(eligible) else validation.iloc[0]
    p = candidates[int(selected_row.id)]
    signal, coefficients = cached[(p.train_window, p.horizon, p.ridge, p.features)]
    signal.to_csv(OUT / "dynamic_signals.csv")
    coefficients.to_csv(OUT / "monthly_model_fits.csv", index=False)
    sizes = []
    for cap in (0., .05, .1, .2, .35, .5, .75, 1.):
        tr = simulate_dynamic(market, signal, p, *PERIODS["training"], cap=cap)[0]
        va = simulate_dynamic(market, signal, p, *PERIODS["validation"], cap=cap)[0]
        st = simulate_dynamic(market, signal, p, *PERIODS["validation"], cap=cap, costs=STRESS)[0]
        valid = bool(len(eligible) and cap > 0 and min(tr["return"], va["return"], st["return"]) > 0
                     and min(tr["drawdown"], va["drawdown"], st["drawdown"]) >= -.05)
        sizes.append({"cap": cap, "initial_gross_cap": cap*100000, "train_return": tr["return"],
                      "validation_return": va["return"], "validation_stress_return": st["return"],
                      "validation_drawdown": va["drawdown"], "eligible": valid})
    sizes = pd.DataFrame(sizes)
    sizes.to_csv(OUT / "sizing.csv", index=False)
    allowed = sizes.loc[sizes.eligible]
    selected_cap = float(allowed.sort_values("validation_return", ascending=False).iloc[0].cap) if len(allowed) else 0.
    evaluation = {}
    for stage, dates in PERIODS.items():
        for name, costs in (("base", BASE), ("stress", STRESS)):
            result = simulate_dynamic(market, signal, p, *dates, costs=costs)
            evaluation[stage+"_"+name] = result[0]
            save_result(stage+"_"+name, result)
    promoted = selected_cap > 0
    promotion_results = {}
    for stage in ("audit", "research_2026"):
        for name, costs in (("base", BASE), ("stress", STRESS)):
            r = simulate_dynamic(market, signal, p, *PERIODS[stage], cap=selected_cap, costs=costs)
            promotion_results[stage+"_"+name] = r[0]
            promoted = promoted and r[0]["return"] > 0 and r[0]["drawdown"] >= -.05 and r[0]["entries"] >= 5
    dynamic = simulate_dynamic(market, signal, p, "2020-01-01", "2026-09-18")
    stressed = simulate_dynamic(market, signal, p, "2020-01-01", "2026-09-18", costs=STRESS)
    save_result("continuous_dynamic", dynamic)
    save_result("continuous_dynamic_stress", stressed)
    equal_params = replace(p, hedge_mode="equal")
    equal_signals, _ = dynamic_signals(market, equal_params)
    equal_result = simulate_dynamic(market, equal_signals, equal_params, "2020-01-01", "2026-09-18")
    save_result("continuous_equal_dollar_ablation", equal_result)
    latest = signal.iloc[-1]
    latest_market = market.iloc[-1]
    tx, ty, threshold, fraction = desired_position(100000, latest_market.X, latest_market.Y,
                                                  latest, p, BASE, selected_cap or .2)
    info = {"parameters": asdict(p), "candidate_count": len(candidates), "period_results": evaluation,
            "selected_validation_cap": selected_cap, "supported_cap_after_audit": selected_cap if promoted else 0.,
            "promotion_results": promotion_results, "continuous_dynamic": dynamic[0],
            "continuous_dynamic_stress": stressed[0],
            "continuous_equal_dollar_ablation": equal_result[0],
            "fixed_order_double_cost_returns": {
                key: value["return"]-(value["commission"]+value["slippage"]+value["borrow"])/100000
                for key, value in evaluation.items() if key.endswith("_base")},
            "mean_gross_dollars": {"dynamic": float(dynamic[1].gross.mean()),
                                   "dynamic_stress": float(stressed[1].gross.mean())},
            "latest": {"date": str(signal.index[-1].date()), "EWA_price": latest_market.X,
                       "EWC_price": latest_market.Y, "EWA_dollar_weight": latest.weight_x,
                       "EWC_dollar_weight": latest.weight_y,
                       "EWA_shares_per_EWC_share": latest.share_ratio_x_per_y,
                       "prediction": latest.prediction, "entry_threshold": threshold,
                       "diagnostic_desired_fraction": fraction, "diagnostic_EWA_qty": tx, "diagnostic_EWC_qty": ty}}
    (OUT / "evaluation.json").write_text(json.dumps(info, indent=2, allow_nan=False))
    make_charts(market, signal, dynamic, stressed, evaluation, sizes)
    lines = ["# 动态信号与对冲比例研究", "",
             "本轮沿用 EWA/EWC 和 10 万美元虚拟账户。所有历史区间均已被观察，本报告是时间顺序研究评估，不能宣称全新样本外检验。", "",
             f"搜索 {len(candidates)} 个线性预测器配置；训练前 12 名进入验证。表中按 20% 最大开仓总金额比较，信号强度及风险预算会降低实际仓位。", "",
             "| 阶段 | 默认成本收益 | 双倍成本重新决策 | 固定原订单双倍成本 | 默认成本回撤 | 开仓 |",
             "|---|---:|---:|---:|---:|---:|"]
    for stage in PERIODS:
        b, s = evaluation[stage+"_base"], evaluation[stage+"_stress"]
        fixed = info["fixed_order_double_cost_returns"][stage+"_base"]
        lines.append(f"| {stage} | {b['return']:.3%} | {s['return']:.3%} | {fixed:.3%} | {abs(b['drawdown']):.3%} | {b['entries']} |")
    lines += ["", f"验证阶段仓位上限选择：{selected_cap:.0%}；经后续历史复核支持的上限：{info['supported_cap_after_audit']:.0%}。零代表未通过盈利/风险门槛。", "",
              "双倍成本重新决策会改变入场门槛和交易次数，不能视为相同订单的纯费用对比；固定订单列保持原股数与成交路径，仅加倍费用和借券。", "",
              "## 连续回测与归因检查", "",
              "2020-01-01 至 2026-09-18，均从 $100,000 开始。", "",
              "| 版本 | 累计收益 | 最大回撤 | 开仓 |", "|---|---:|---:|---:|"]
    for name, result in (("动态信号＋动态对冲", dynamic), ("本版双倍成本重新决策", stressed),
                         ("本版等金额对冲消融", equal_result)):
        r = result[0]
        lines.append(f"| {name} | {r['return']:.3%} | {abs(r['drawdown']):.3%} | {r['entries']} |")
    lines += ["", f"默认成本平均实际总敞口 ${info['mean_gross_dollars']['dynamic']:,.0f}。小回撤部分来自少交易和低实际仓位，并非纯粹预测能力的证明。",
              "等金额消融只改本版对冲方式并重新训练对应目标，不按消融收益重新选参数。",
              "本轮先测试 48 组一般线性预测器，再加入 24 组 Kalman-z 对称回归预测器；这是探索式研究，不能把迭代后的表现当作独立检验。", "",
              "## 资金比例", "",
              f"截至 {info['latest']['date']}：EWA 占双腿总名义金额 {latest.weight_x:.2%}，EWC 占 {latest.weight_y:.2%}；股数比例 |q_EWA|/|q_EWC|={latest.share_ratio_x_per_y:.4f}。一腿买入、另一腿卖空，方向由预测决定。",
              "该比例是在固定总金额、两腿相反方向、40%–60% 权重限制下，最小化估计组合方差及权重变动惩罚的条件最优。它不是收益最大化保证，也不是市场 Beta 中性证明。",
              "这是双腿名义金额比例，不是券商保证金或实际现金占用比例。以总名义金额 $20,000 为例约为 EWA $8,295、EWC $11,705；当前信号未超过费用门槛，因此目标股数为零。",
              "设 EWA 权重 a，目标 J=a²vx+(1-a)²vy−2a(1-a)c+λ(a−0.5)²+η(a−a_prev)²。",
              "a=clip((vy+c+λ/2+ηa_prev)/(vx+vy+2c+λ+η), 0.4, 0.6)，λ=0.1(vx+vy)，η=0.5(vx+vy)。",
              "q_EWA≈−direction×G×a/P_EWA，q_EWC≈direction×G×(1−a)/P_EWC，取整数。", "",
              "## 动态信号", "",
              "每月用过去 252 或 504 日可用样本重训岭回归，每天根据当前特征更新预测。预测标签从下一日收盘开始，覆盖 5 或 10 个交易日；只有已完全到期的标签才能参与训练。",
              "Kalman-z 版本保留慢速 Kalman 的价差特征，学习对称回归幅度：截距固定为零，z 的系数限制为非正；若历史不支持回归，系数可以为零，不交易。一般版本使用价差 z、变化和多周期相对收益。",
              "阈值=预计往返佣金/滑点/借券成本÷总金额＋置信门槛×近期日波动×sqrt(预测天数)。小仓位取整后再次检查最低费用。",
              "波动预算限制账户目标年化波动为 2%，仓位同时受信号强度、资金上限和相关性过滤影响。持仓比例偏离超过 4 个百分点且风险改善超过再平衡费才再平衡。",
              "信号日固定整数股数，下一日收盘模拟成交；跳空导致开仓总金额超上限时取消该笔开仓。期末统一平仓计费。",
              "5% 回撤触发后，下一日退出并停止该次回测的新交易；跳空/延迟意味着实际回撤仍可能超过 5%。", "",
              "## 成本与限制", "",
              "每股 $0.005、每腿每单最低 $1，滑点 2 bp、卖出费用预留 0.3 bp，借券年化 3% 按日历天计。压力测试全部加倍。分红单独入账，含拆股样本拒绝。",
              "尚未包含真实借券可用性、双腿异步成交、盘口冲击和完整税费。当前最优比例可计算，但配置通过历史筛选也不证明未来盈利。",
              "连续曲线从 2020 年以 10 万美元起步，中间不重置；各阶段独立表则从平仓状态和 10 万美元重新起算，两者不可混用。", "",
              "## 复现和文件", "", "运行 `python scripts/evaluate_dynamic.py`。",
              "dynamic_signals.csv 是每日权重、预测与风险估计；monthly_model_fits.csv 记录拟合时点与最后标签到期日；各场景下提供权益、订单事件及费用摘要。",
              "performance.png/svg 和 validation_and_sizing.png/svg 为图像；backtest_bundle.zip 为完整派生数据及图像包。", "",
              "方法参考：[Ridge](https://scikit-learn.org/1.8/modules/generated/sklearn.linear_model.Ridge.html)、[时间序列验证](https://scikit-learn.org/1.4/modules/generated/sklearn.model_selection.TimeSeriesSplit.html)。", ""]
    (OUT / "REPORT.md").write_text("\n".join(lines))
    with zipfile.ZipFile(OUT / "backtest_bundle.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for path in OUT.rglob("*"):
            if path.is_file() and path.suffix in (".csv", ".json", ".md", ".png", ".svg") and ".mpl-cache" not in path.parts:
                archive.write(path, path.relative_to(OUT))
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()

"""Predeclared multi-pair chronological research with segregated cash sleeves."""
import json
import os
import shutil
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd

from pairs_trading.dynamic import DynamicParameters, dynamic_signals, simulate_dynamic
from pairs_trading.research import Costs, load_market

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/multi_pair/results"
DATA = ROOT / "output/multi_pair/market_data"
PAIRS = [("EWA", "EWC"), ("XLE", "VDE"), ("XLF", "VFH"),
         ("XLP", "VDC"), ("XLV", "VHT"), ("XLI", "VIS")]
CAPITAL = 100000
SLEEVE = 25000  # Four slots; unused slots remain cash, never redistributed after test results.
BASE = Costs()
STRESS = replace(BASE, commission_per_share=.01, minimum_per_order=2,
                 slippage_bps=4, sell_fee_bps=.6, borrow_annual=.06)
PERIODS = {"train": ("2017-01-01", "2019-12-31"),
           "validation": ("2020-01-01", "2022-12-31"),
           "audit": ("2023-01-01", "2026-09-18")}


def save_result(name, result):
    folder = OUT / name
    folder.mkdir(parents=True, exist_ok=True)
    summary, curve, trades = result
    (folder / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    curve.to_csv(folder / "equity.csv")
    trades.to_csv(folder / "trades.csv", index=False)


def portfolio(results, index):
    if len(results) > 4:
        raise ValueError("more sleeves than the account budget permits")
    curve = pd.DataFrame(index=index)
    curve["equity"] = float(CAPITAL)
    curve["gross"] = 0.
    curve["net"] = 0.
    for summary, leg, trades in results:
        aligned = leg.reindex(index)
        if aligned.equity.isna().any():
            raise ValueError("unaligned sleeve calendar; do not forward fill marked positions")
        curve["equity"] += aligned.equity - SLEEVE
        curve["gross"] += aligned.gross
        curve["net"] += aligned.net
    returns = curve.equity.pct_change().fillna(0)
    vol = returns.std()*np.sqrt(252)
    summary = {"initial_equity": CAPITAL, "final_equity": float(curve.equity.iloc[-1]),
               "return": float(curve.equity.iloc[-1]/CAPITAL-1),
               "drawdown": float((curve.equity/curve.equity.cummax()-1).min()),
               "sharpe": float(returns.mean()*252/vol) if vol else 0.,
               "entries": sum(r[0]["entries"] for r in results),
               "commission": sum(r[0]["commission"] for r in results),
               "slippage": sum(r[0]["slippage"] for r in results),
               "borrow": sum(r[0]["borrow"] for r in results),
               "mean_gross": float(curve.gross.mean()), "max_gross": float(curve.gross.max())}
    return summary, curve


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True, exist_ok=True)
    params = [DynamicParameters(train_window=504, horizon=h, ridge=.1,
                               confidence=.1, features=f)
              for h in (5, 10) for f in ("simple", "full", "kalman")]
    protocol = {"pairs": PAIRS, "parameters": [asdict(p) for p in params],
                "periods": PERIODS, "capital_usd": CAPITAL, "sleeve_usd": SLEEVE,
                "max_slots": 4, "costs": asdict(BASE), "stress_costs": asdict(STRESS),
                "selection": "positive train/validation/stress validation, >=8 entries train and validation, <=5% drawdown each; rank min(train,validation) Sharpe; one model per pair; no shared symbols",
                "audit_usage": "Never used for selection in this script; retrospective universe/model development is not an untouched holdout.",
                "provider": "Yahoo Finance via yfinance, not Alpaca",
                "risk": "Sleeves capped at 100% entry gross, sum <= account capital at entry; market moves can exceed cap. Each sleeve stops after 5% drawdown, next-close exit, not a guaranteed loss bound.",
                "limitations": ["Surviving ETF universe; not a point-in-time fund universe.",
                                "Minimum variance is not market-beta neutrality.",
                                "Borrow assumed available; no locate rejection or volume participation cap.",
                                "Daily close simulation, not HFT or a running paper account."]}
    (OUT / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    rows, excluded, cache, markets = [], [], {}, {}
    for pair in PAIRS:
        name = "_".join(pair)
        try:
            market = load_market(DATA, pair)
        except ValueError as exc:
            excluded.append({"pair": name, "reason": str(exc)})
            continue
        markets[name] = market
        for number, p in enumerate(params):
            signals, fits = dynamic_signals(market, p)
            cache[(name, number)] = signals, fits
            row = {"pair": name, "parameter_id": number, **asdict(p)}
            for phase, costs in (("train", BASE), ("validation", BASE), ("stress", STRESS)):
                dates = PERIODS["validation" if phase == "stress" else phase]
                result = simulate_dynamic(market, signals, p, *dates, cap=1,
                                          capital=SLEEVE, costs=costs)
                row.update({phase+"_"+k: v for k, v in result[0].items()})
            row["eligible"] = bool(all(row[phase+"_return"] > 0 and row[phase+"_drawdown"] >= -.05
                                       for phase in ("train", "validation", "stress"))
                                   and row["train_entries"] >= 8 and row["validation_entries"] >= 8)
            row["score"] = min(row["train_sharpe"], row["validation_sharpe"])
            rows.append(row)
        print(name, "evaluated", flush=True)
    search = pd.DataFrame(rows)
    if search.empty:
        raise ValueError("no supported candidate pairs")
    search.to_csv(OUT / "search.csv", index=False)
    eligible = search[search.eligible].sort_values("score", ascending=False)
    selected, used = [], set()
    for _, row in eligible.iterrows():
        symbols = set(row.pair.split("_"))
        if not symbols & used:
            selected.append({"pair": row.pair, "parameter_id": int(row.parameter_id)})
            used |= symbols
        if len(selected) == 4:
            break
    (OUT / "selection.json").write_text(json.dumps({"selected": selected, "excluded": excluded,
        "supported_capital_usd": len(selected)*SLEEVE, "cash_reserve_usd": CAPITAL-len(selected)*SLEEVE}, indent=2)+"\n")
    # Audit only the already frozen selection. Diagnostic candidates are explicitly not promoted.
    results, stress_results, weights = [], [], []
    for chosen in selected:
        name, number = chosen["pair"], chosen["parameter_id"]
        signals, fits = cache[(name, number)]
        market, p = markets[name], params[number]
        for suffix, costs, destination in (("base", BASE, results), ("stress", STRESS, stress_results)):
            r = simulate_dynamic(market, signals, p, *PERIODS["audit"], cap=1,
                                 capital=SLEEVE, costs=costs)
            destination.append(r)
            save_result(name+"_"+suffix, r)
        signals.to_csv(OUT / f"{name}_signals.csv")
        fits.to_csv(OUT / f"{name}_fits.csv", index=False)
        latest = signals.iloc[-1]
        weights.append({"pair": name, "date": str(signals.index[-1].date()),
                        "weight_x": latest.weight_x, "weight_y": latest.weight_y,
                        "share_ratio": latest.share_ratio_x_per_y, "prediction": latest.prediction,
                        "sleeve_capital": SLEEVE})
    # Also expose conditional hedge ratios for rejected candidates; allocation remains zero.
    for _, candidate in search.sort_values("score", ascending=False).drop_duplicates("pair").iterrows():
        if any(w["pair"] == candidate.pair for w in weights):
            continue
        signals, _ = cache[(candidate.pair, int(candidate.parameter_id))]
        latest = signals.iloc[-1]
        weights.append({"pair": candidate.pair, "date": str(signals.index[-1].date()),
                        "weight_x": latest.weight_x, "weight_y": latest.weight_y,
                        "share_ratio": latest.share_ratio_x_per_y, "prediction": latest.prediction,
                        "sleeve_capital": 0})
    index = next(iter(markets.values())).loc[slice(*PERIODS["audit"])].index
    summary, curve = portfolio(results, index)
    stressed, stress_curve = portfolio(stress_results, index)
    trade_frames = [r[2].assign(pair=chosen["pair"]) for chosen, r in zip(selected, results)]
    trades = pd.concat(trade_frames, ignore_index=True) if trade_frames else pd.DataFrame(
        columns=["pair", "date", "signal_date", "event", "delta_qx", "delta_qy", "qx", "qy",
                 "px", "py", "commission", "slippage"])
    trades.to_csv(OUT / "trades.csv", index=False)
    curve.to_csv(OUT / "equity.csv")
    stress_curve.to_csv(OUT / "stress_equity.csv")
    pd.DataFrame(weights, columns=["pair", "date", "weight_x", "weight_y", "share_ratio", "prediction", "sleeve_capital"]).to_csv(OUT / "latest_hedges.csv", index=False)
    evaluation = {"base": summary, "stress_filter_rerun": stressed,
                  "fixed_order_double_cost_return": summary["return"]-(summary["commission"]+summary["slippage"]+summary["borrow"])/CAPITAL,
                  "selected": selected, "excluded": excluded,
                  "status": "research_only_not_paper_validated"}
    (OUT / "evaluation.json").write_text(json.dumps(evaluation, indent=2)+"\n")
    os.environ.setdefault("MPLCONFIGDIR", str(OUT / ".mpl-cache"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("Multi-pair ETF research · $100,000 shared budget", fontsize=18)
    if not selected:
        fig.text(.5, .94, "NO QUALIFYING PAIRS — account stays in cash; zero return is not trading alpha", ha="center", color="#b33c35")
    for label, frame in (("Base costs", curve), ("Doubled costs; filters rerun", stress_curve)):
        axes[0, 0].plot(frame.index, frame.equity/CAPITAL-1, label=label)
        axes[0, 1].plot(frame.index, frame.equity/frame.equity.cummax()-1, label=label)
    for ax in axes[0]:
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.legend(fontsize=8)
        ax.grid(alpha=.2)
    axes[0, 0].set_title("2023–2026 retrospective audit return")
    axes[0, 1].set_title("Account drawdown")
    best = search.sort_values("score", ascending=False).drop_duplicates("pair").set_index("pair")
    best[["train_return", "validation_return", "stress_return"]].plot.bar(ax=axes[1, 0])
    axes[1, 0].set_title("Best development candidate per pair · $25k sleeve")
    axes[1, 0].yaxis.set_major_formatter(PercentFormatter(1))
    axes[1, 0].tick_params(axis="x", rotation=15)
    axes[1, 0].legend(fontsize=7)
    axes[1, 1].plot(curve.index, curve.gross/CAPITAL)
    axes[1, 1].yaxis.set_major_formatter(PercentFormatter(1))
    axes[1, 1].set_title("Actual gross exposure / starting capital")
    fig.text(.05, .015, "Yahoo daily data; not Alpaca. Frozen chronological selection; retrospective research, not an untouched test. No paper orders.", fontsize=9)
    fig.tight_layout(rect=(0, .04, 1, .95))
    for extension in ("png", "svg"):
        fig.savefig(OUT / ("performance."+extension), dpi=160)
    plt.close(fig)
    for target in OUT.glob("*.svg"):
        target.write_text("\n".join(line.rstrip() for line in target.read_text().splitlines())+"\n")
    selected_text = ", ".join(x["pair"] for x in selected) or "无；所有资金保持现金"
    report = f'''# 多交易对动态策略回测

数据：Yahoo Finance 免费真实日线，2015–2026；尚未配置 Alpaca key，因此本结果不是 Alpaca 回测。

- 候选：6 组、12 个 ETF；可用 {len(markets)} 组，拆股不支持的配对明确排除。
- 每组 6 个参数配置；训练 2017–2019，验证 2020–2022，随后冻结组合。
- 入选：{selected_text}。每组独立 25,000 美元资金切片，总账户 100,000 美元。
- 后续审计：2023–2026-09-18，费用后收益 **{summary['return']:.3%}**，最大回撤 **{abs(summary['drawdown']):.3%}**，入场 {summary['entries']} 次。
- 双倍成本重新运行信号过滤：{stressed['return']:.3%}；固定成交序列成本加倍估计：{evaluation['fixed_order_double_cost_return']:.3%}。
- 期末权益 ${summary['final_equity']:,.2f}；平均名义敞口 ${summary['mean_gross']:,.2f}。

![performance](performance.png)

## 模型与资金

沿用因果月度线性 ridge 预测和滚动协方差最小方差 hedge。
比较 5/10 日预测、simple/full/Kalman 三种特征，固定 ridge=0.1 和 confidence=0.1。
预测训练只使用已到期标签，信号收盘确定股数、次日收盘执行；买卖两腿、再平衡、
股息、日历日借券和最低佣金均进入账本。阈值动态覆盖成本与预测风险。
训练/验证/双倍成本验证均盈利且训练与验证各不少于 8 次入场才可入选。
按训练与验证中较低的 Sharpe 排序，一对只选一个模型，最多四对且没有共享标的。
未用资金为现金、利息为零。没有按审计期利润重新分配资金，也没有把独立账户收益直接相加。
参数及资金仅为这组实验的约束内选择，不构成全局最优本金结论。

## 限制

这是事后研究，候选池使用存续 ETF，模型已经在旧研究中开发，不宣称全新未见测试。
行业 ETF 对冲降低共同波动，不保证 beta 中性。尚无借券拒绝、成交量参与限制或真实撮合。
拆股排除、连续日线、收盘同步等数据假设见 protocol.json 和 selection.json。
日线回测不能证明高频性能，模拟盘尚未启动。没有入选时，零收益代表持币，不代表策略已盈利。
旧版与本版评估区间和仓位预算不同，不应直接比较表面累计收益。

[搜索结果](search.csv) · [选择及排除原因](selection.json) · [汇总](evaluation.json) ·
[权益](equity.csv) · [最新 hedge](latest_hedges.csv) · [完整协议](protocol.json)
'''
    (OUT / "REPORT.md").write_text(report)
    print(json.dumps(evaluation, indent=2), flush=True)


if __name__ == "__main__":
    main()

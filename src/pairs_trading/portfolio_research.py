"""Daily portfolio prototype. Retrospective adjusted-price research, never an order engine."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def load_panel(directory, config):
    root = Path(directory)
    meta = json.loads((root / "metadata.json").read_text())
    if meta.get("provider", "").startswith("Yahoo"):
        frames = []
        for i, symbol in enumerate(meta["symbols"]):
            f = pd.read_csv(root / f"raw_{i}.csv")
            f["date"] = pd.to_datetime(f.iloc[:, 0].str[:10])
            f = f.rename(
                columns={
                    "Close": "raw_close_proxy",
                    "Adj Close": "adjusted_close",
                    "Volume": "volume",
                    "Stock Splits": "split",
                    "Dividends": "dividend",
                }
            )
            f["symbol"] = symbol
            frames.append(
                f[
                    [
                        "date",
                        "symbol",
                        "raw_close_proxy",
                        "adjusted_close",
                        "volume",
                        "split",
                        "dividend",
                    ]
                ]
            )
        panel = pd.concat(frames, ignore_index=True)
    else:
        raw = pd.read_csv(root / "bars_raw.csv")
        adj = pd.read_csv(root / "bars_all.csv")
        panel = raw.merge(
            adj[["symbol", "t", "c"]],
            on=["symbol", "t"],
            suffixes=("", "_adj"),
            validate="one_to_one",
        )
        panel["date"] = (
            pd.to_datetime(panel.t, utc=True)
            .dt.tz_convert("America/New_York")
            .dt.tz_localize(None)
            .dt.normalize()
        )
        panel = panel.rename(
            columns={"c": "raw_close_proxy", "c_adj": "adjusted_close", "v": "volume"}
        )
        panel["split"] = np.nan
        panel["dividend"] = np.nan
    if panel.duplicated(["date", "symbol"]).any():
        raise ValueError("duplicate daily bars")
    values = panel[["raw_close_proxy", "adjusted_close", "volume"]].to_numpy()
    if not np.isfinite(values).all() or (values[:, :2] <= 0).any() or (values[:, 2] < 0).any():
        raise ValueError("invalid price/volume")
    expected = set(config["groups"]) | {config["market"]}
    if set(panel.symbol) != expected:
        raise ValueError("symbol coverage mismatch")
    return panel.sort_values(["date", "symbol"]), meta


def build_features(panel, config):
    def wide(field):
        return panel.pivot(index="date", columns="symbol", values=field).sort_index()

    price = wide("adjusted_close")
    dates = price.index[price[config["market"]].notna()]
    price = price.reindex(dates)
    raw, volume = wide("raw_close_proxy").reindex(dates), wide("volume").reindex(dates)
    returns = price.pct_change(fill_method=None)
    symbols = list(config["groups"])
    r = returns[symbols]
    market = returns[config["market"]]
    beta = (
        r.rolling(126, min_periods=126).cov(market).div(market.rolling(126).var(), axis=0).shift(1)
    )
    mean = r.rolling(126).mean().shift(1)
    mmean = market.rolling(126).mean().shift(1)
    residual = r - mean - beta.mul(market - mmean, axis=0)
    for sector in sorted(set(config["groups"].values())):
        cols = [s for s in symbols if config["groups"][s] == sector]
        # Second, lagged regression on the contemporaneous group's market residual.
        group = residual[cols].mean(axis=1)
        gamma = (
            residual[cols].rolling(126).cov(group).div(group.rolling(126).var(), axis=0).shift(1)
        )
        residual.loc[:, cols] = residual[cols] - gamma.mul(group, axis=0)
    variance = r.rolling(60, min_periods=40).var()
    adv = (raw[symbols] * volume[symbols]).rolling(60, min_periods=57).median()
    coverage = price[symbols].notna().rolling(60, min_periods=60).mean()
    stale = raw[symbols].diff().eq(0).rolling(5).sum().ge(5)
    eligible = (
        (raw[symbols] >= 5)
        & (price[symbols].notna().cumsum() >= 252)
        & (coverage >= 0.95)
        & (adv >= 1e6)
        & ~stale
        & (volume[symbols] > 0)
    )
    eligible &= adv.where(eligible).rank(axis=1, ascending=False, method="first") <= 400
    # Scalar market beta Kalman prior; no posterior residual look-ahead.
    kb, uncertainty = np.ones(len(symbols)), np.ones(len(symbols))
    innovations = pd.DataFrame(np.nan, index=r.index, columns=symbols)
    for date in r.index:
        x, y = market.loc[date], r.loc[date].to_numpy()
        if not np.isfinite(x):
            continue
        valid = np.isfinite(y)
        uncertainty += 1e-5
        e = y - kb * x
        innovations.loc[date] = e
        gain = uncertainty * x / (1e-4 + x * x * uncertainty)
        kb[valid] += gain[valid] * e[valid]
        uncertainty[valid] *= 1 - gain[valid] * x
    for sector in set(config["groups"].values()):
        cols = [s for s in symbols if config["groups"][s] == sector]
        innovations.loc[:, cols] = innovations[cols].sub(innovations[cols].mean(axis=1), axis=0)
    surprise = (volume[symbols] / volume[symbols].rolling(60).median()).clip(0, 5)
    signals = {
        "reversal": -residual.rolling(3).sum(),
        "kalman": -innovations.rolling(3).sum(),
        "dislocation": -residual * surprise / np.sqrt(variance),
    }

    def zscore(f):
        f = f.where(eligible)
        return f.sub(f.mean(axis=1), axis=0).div(f.std(axis=1), axis=0).clip(-3, 3)

    signals = {k: zscore(v) for k, v in signals.items()}
    signals["blend"] = sum(signals.values()) / 3
    for horizon in [2, 5]:
        signals[f"blend_h{horizon}"] = (
            zscore(-residual.rolling(horizon).sum()) + signals["kalman"] + signals["dislocation"]
        ) / 3
    return {
        "returns": r,
        "beta": beta,
        "variance": variance,
        "adv": adv,
        "eligible": eligible,
        "signals": signals,
        "price": price[symbols],
        "coverage": coverage,
        "stale": stale,
    }


def neutral_target(score, beta, variance, groups):
    valid = np.isfinite(score) & np.isfinite(beta) & np.isfinite(variance) & (variance > 0)
    idx = np.flatnonzero(valid)
    out = np.zeros(len(score))
    if len(idx) < 4:
        return out
    g = np.array(groups)[idx]
    exposure = np.column_stack([g == sector for sector in sorted(set(g))] + [beta[idx]])
    a = score[idx] / np.maximum(variance[idx], 1e-6)
    a -= exposure @ np.linalg.lstsq(exposure, a, rcond=None)[0]
    gross = np.abs(a).sum()
    if gross < 1e-8:
        return out
    a /= gross
    a *= min(1.0, 0.10 / max(np.abs(a)))
    out[idx] = a
    return out


def trading_cost(delta, vol, adv, nav, multiplier=1.0):
    q = np.abs(delta) * nav
    # A missing ADV on a required trade is not zero-cost liquidity.
    if ((q > 1e-7) & (~np.isfinite(adv) | (adv <= 0) | ~np.isfinite(vol))).any():
        raise ValueError("missing cost inputs for traded asset")
    impact = 0.10 * np.nan_to_num(vol) * np.sqrt(q / np.maximum(np.nan_to_num(adv), 1.0))
    return multiplier * np.abs(delta) * (0.00005 + 0.0002 + 0.0001 + impact)


def simulate(
    features,
    config,
    family="blend",
    start="2023-01-01",
    end="2024-12-31",
    multiplier=1.0,
    delay=1,
    frequency=1,
    excluded=(),
):
    r = features["returns"]
    dates = r.loc[start:end].index
    groups = list(config["groups"].values())
    symbols = list(config["groups"])
    w = np.zeros(len(symbols))
    nav = 100000.0
    rows, weights, contributions = [], [], []
    for k, date in enumerate(dates):
        pos = r.index.get_loc(date)
        if pos < delay + 1:
            continue
        y = r.loc[date].to_numpy()
        if ((np.abs(w) > 1e-10) & ~np.isfinite(y)).any():
            raise ValueError(f"missing held return {date}")
        y = np.nan_to_num(y)
        gross_components = w * y
        gross = gross_components.sum()
        if 1 + gross <= 0:
            raise ValueError("insolvent portfolio")
        drift = w * (1 + y) / (1 + gross)
        pretrade_nav = nav * (1 + gross)
        decision = r.index[pos - delay]
        score = features["signals"][family].loc[decision].copy()
        score.loc[list(excluded)] = np.nan
        beta = features["beta"].loc[decision].to_numpy()
        var = features["variance"].loc[decision].to_numpy()
        adv = features["adv"].loc[decision].to_numpy()
        desired = neutral_target(score.to_numpy(), beta, var, groups)
        days = (date - dates[k - 1]).days if k else 1
        borrow_by_name = np.maximum(-w, 0) * 0.03 * days / 365 * multiplier
        borrow = borrow_by_name.sum()
        override = False
        if k % frequency != 0:
            target = drift.copy()
        else:
            # Reproject drift so blending preserves exact current neutrality.
            valid = np.isfinite(score.to_numpy()) & np.isfinite(beta) & np.isfinite(var)
            idx = np.flatnonzero(valid)
            anchor = np.zeros(len(w))
            if len(idx):
                g = np.array(groups)[idx]
                x = np.column_stack([g == s for s in sorted(set(g))] + [beta[idx]])
                anchor[idx] = drift[idx] - x @ np.linalg.lstsq(x, drift[idx], rcond=None)[0]

            def feasible(
                target,
                drift=drift,
                adv=adv,
                pretrade_nav=pretrade_nav,
                var=var,
                gross=gross,
                borrow=borrow,
            ):
                d = np.abs(target - drift)
                liquidity = np.nan_to_num(adv) * 0.001 / pretrade_nav
                cost_est = trading_cost(
                    target - drift, np.sqrt(var), adv, pretrade_nav, multiplier
                ).sum()
                ratio = (1 + gross) / ((1 + gross) * (1 - cost_est) - borrow)
                return (
                    d.sum() <= 0.5 + 1e-9
                    and (d <= liquidity + 1e-9).all()
                    and np.abs(target).sum() * ratio <= 1 + 1e-9
                    and np.abs(target).max() * ratio <= 0.1 + 1e-9
                )

            target = None
            for mix in np.linspace(1, 0, 51):
                proposed = anchor + mix * (desired - anchor)
                if feasible(proposed):
                    target = proposed
                    break
            if target is None:
                target = np.zeros(len(w))
                override = True
        if k == len(dates) - 1:
            target = np.zeros(len(w))
        delta = target - drift
        cost_by_name = trading_cost(delta, np.sqrt(var), adv, pretrade_nav, multiplier)
        cost = cost_by_name.sum() * (1 + gross)
        borrow = borrow_by_name.sum()
        net = gross - cost - borrow
        nav *= 1 + net
        # Costs deducted from cash: actual post-fee weights differ by this factor.
        postfee = (1 + gross) / (1 + net)
        w = target * postfee
        sector = max(abs(w[np.array(groups) == s].sum()) for s in set(groups))
        beta_exp = float(w @ np.nan_to_num(beta))
        ic = np.nan
        # Diagnostic signal vs first earned forward return (not used for selection here).
        if pos + 1 < len(r) and r.index[pos + 1] <= dates[-1]:
            nextret = r.iloc[pos + 1]
            valid_ic = score.notna() & nextret.notna()
            if valid_ic.sum() >= 4 and score[valid_ic].std() > 0 and nextret[valid_ic].std() > 0:
                ic = score[valid_ic].rank().corr(nextret[valid_ic].rank())
        rows.append(
            {
                "date": date,
                "net_return": net,
                "gross_return": gross,
                "transaction_cost": cost,
                "borrow_cost": borrow,
                "nav": nav,
                "turnover": np.abs(delta).sum(),
                "gross_exposure": np.abs(w).sum(),
                "net_exposure": w.sum(),
                "beta_exposure": beta_exp,
                "max_group_exposure": sector,
                "max_weight": np.abs(w).max(),
                "rank_ic": ic,
                "risk_liquidation": override,
                "capacity_proxy_nav": float(
                    np.min(
                        np.nan_to_num(adv)[np.abs(delta) > 1e-8]
                        * 0.001
                        / np.abs(delta)[np.abs(delta) > 1e-8]
                    )
                )
                if (np.abs(delta) > 1e-8).any()
                else np.nan,
            }
        )
        weights.append(w.copy())
        contributions.append(gross_components - cost_by_name * (1 + gross) - borrow_by_name)
    frame = pd.DataFrame(rows).set_index("date")
    return (
        frame,
        pd.DataFrame(weights, index=frame.index, columns=symbols),
        pd.DataFrame(contributions, index=frame.index, columns=symbols),
    )


def metrics(f):
    r = f.net_return
    std = r.std()
    curve = (1 + r).cumprod()
    high = curve.cummax().clip(lower=1.0)
    yearly = r.groupby(r.index.year).apply(lambda v: float((1 + v).prod() - 1)).to_dict()
    return {
        "sessions": len(f),
        "net_total_return": float(curve.iloc[-1] - 1),
        "sharpe": float(r.mean() / std * np.sqrt(252)) if std > 0 else 0.0,
        "max_drawdown": float((curve / high - 1).min()),
        "mean_rank_ic": float(f.rank_ic.mean()),
        "mean_daily_turnover": float(f.turnover.mean()),
        "hit_rate": float((r > 0).mean()),
        "transaction_cost_sum": float(f.transaction_cost.sum()),
        "borrow_cost_sum": float(f.borrow_cost.sum()),
        "yearly_returns": yearly,
        "max_abs_net": float(f.net_exposure.abs().max()),
        "max_abs_beta": float(f.beta_exposure.abs().max()),
        "max_abs_group": float(f.max_group_exposure.max()),
        "risk_liquidations": int(f.risk_liquidation.sum()),
    }


def accepted(m):
    full_years = [v for year, v in m["yearly_returns"].items() if int(year) < 2026]
    return (
        m["sessions"] >= 252
        and m["sharpe"] >= 1
        and m["net_total_return"] > 0
        and m["max_drawdown"] >= -0.1
        and m["mean_rank_ic"] > 0
        and all(v > 0 for v in full_years)
        and m["risk_liquidations"] == 0
    )


def write_research(data, output, config, protocol):
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    panel, source = load_panel(data, config)
    features = build_features(panel, config)
    panel.to_csv(root / "daily_panel.csv.gz", index=False)
    features["eligible"].to_csv(root / "universe_membership.csv.gz")
    pd.DataFrame(
        {
            "members": features["eligible"].sum(axis=1),
            "stale_names": features["stale"].sum(axis=1),
            "missing_prices": features["price"].isna().sum(axis=1),
        }
    ).to_csv(root / "data_quality.csv")
    summary, base, attribution = {}, {}, {}
    periods = {
        "train": ("2018-01-01", "2022-12-31"),
        "validation": ("2023-01-01", "2024-12-31"),
        "audit": ("2025-01-01", "2026-09-18"),
    }
    for family in ["reversal", "kalman", "dislocation", "blend"]:
        summary[family] = {}
        for name, (start, end) in periods.items():
            f, w, a = simulate(features, config, family, start, end)
            summary[family][name] = metrics(f)
            f.to_csv(root / f"{family}_{name}.csv")
            if family == "blend":
                base[name], attribution[name] = f, a
                w.to_csv(root / f"weights_{name}.csv.gz")
                a.to_csv(root / f"attribution_{name}.csv.gz")
    best = list(
        attribution["validation"]
        .sum()
        .nlargest(max(1, int(np.ceil(len(config["groups"]) * 0.05))))
        .index
    )
    removed = list(config["groups"])[::5]
    stress = {}
    scenarios = {
        "double_cost": {"multiplier": 2},
        "delay_one_more": {"delay": 2},
        "horizon_2": {"family": "blend_h2"},
        "horizon_5": {"family": "blend_h5"},
        "alternate_days": {"frequency": 2},
        "remove_top_5pct": {"excluded": best},
        "remove_20pct": {"excluded": removed},
    }
    for scenario, opts in scenarios.items():
        stress[scenario] = {}
        for name in ["validation", "audit"]:
            start, end = periods[name]
            f, _, _ = simulate(features, config, start=start, end=end, **opts)
            stress[scenario][name] = metrics(f)
    statistical_pass = all(accepted(summary["blend"][p]) for p in ["validation", "audit"])
    stress_pass = all(
        m["net_total_return"] > 0 and m["risk_liquidations"] == 0
        for s in stress.values()
        for m in s.values()
    )
    reasons = [
        "Historical PIT security master and delisting reconciliation unavailable",
        "Retrospective adjusted prices; raw execution/action reconciliation incomplete",
        "Borrow availability and observed spreads/fills unavailable",
        "ETF prototype; 200–500-stock universe not collected",
        "2025–2026 already inspected in previous project: not blind OOS",
    ]
    result = {
        "status": "REJECTED_RESEARCH_ONLY_CASH",
        "statistical_pass": statistical_pass,
        "stress_pass": stress_pass,
        "promotion_blockers": reasons,
        "metrics": summary,
        "stress": stress,
        "best_validation_names_removed": best,
        "protocol_sha256": hashlib.sha256(Path(protocol).read_bytes()).hexdigest(),
        "source": source,
        "paper_orders_submitted": 0,
    }
    (root / "evaluation.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    date = features["returns"].index[-1]
    shadow = neutral_target(
        features["signals"]["blend"].loc[date].to_numpy(),
        features["beta"].loc[date].to_numpy(),
        features["variance"].loc[date].to_numpy(),
        list(config["groups"].values()),
    )
    pd.DataFrame(
        {
            "symbol": list(config["groups"]),
            "decision_date": str(date.date()),
            "research_weight": shadow,
            "approved_weight": 0.0,
            "status": "REJECTED_RESEARCH_ONLY_CASH",
        }
    ).to_csv(root / "paper_targets.csv", index=False)
    (root / "protocol.md").write_text(Path(protocol).read_text())
    val, audit = summary["blend"]["validation"], summary["blend"]["audit"]
    provider_label = (
        "Alpaca IEX ETF 数据交叉检查" if source.get("feed") == "iex" else "Yahoo ETF 原型"
    )
    report = f"""# 组合策略研究结果\n\n状态：拒绝晋级，批准仓位为现金；没有提交订单。\n\n数据来源：{provider_label}。\n\n采集 {panel.symbol.nunique()} 个代码，{len(panel):,} 条日线，{panel.date.min().date()} 至 {panel.date.max().date()}。22 个 ETF 为候选，SPY 为基准。\n\n| 区间 | 扣成本收益 | Sharpe | 最大回撤 | 日均换手 |\n|---|---:|---:|---:|---:|\n"""
    for name, m in [("验证 2023–2024", val), ("时间留出审计 2025–2026", audit)]:
        report += f"| {name} | {m['net_total_return']:.2%} | {m['sharpe']:.2f} | {m['max_drawdown']:.2%} | {m['mean_daily_turnover']:.2%} |\n"
    report += "\n固定组合：残差反转 + Kalman 动态相对价值 + 量价偏离，等权混合。按 ETF 经济组与市场 beta 中性化，考虑持仓漂移、下一收盘执行、冲击/价差/佣金/借券成本。\n\n"
    report += f"基础统计门槛通过：{statistical_pass}；全部压力测试通过：{stress_pass}。详细分策略、分年度和成本分解见 evaluation.json；每日收益、仓位及按名称归因见 CSV。\n\n"
    report += "## 压力测试\n\n| 场景 | 验证收益 | 审计收益 |\n|---|---:|---:|\n"
    for name, m in stress.items():
        report += f"| {name} | {m['validation']['net_total_return']:.2%} | {m['audit']['net_total_return']:.2%} |\n"
    report += "\n## 未完成的生产条件\n\n" + "\n".join("- " + s for s in reasons)
    report += "\n\n本次为 ETF 原型，不是美股 400 标的 PIT 回测。IEX 成交量不能解释为全市场 ADV，结果不能与全市场数据直接等同比较。成本为预设代理；不能据此宣称可交易 alpha。market cap/历史行业标签缺失，未补造。paper_targets.csv 中 research_weight 仅供观察，approved_weight 均为 0。下一步需要本机配置 Alpaca 凭据，获得历史证券主表及公司行为数据，并完成真实 paper 成交对账和 2–3 个月前瞻观察。\n"
    (root / "REPORT.md").write_text(report)
    return result

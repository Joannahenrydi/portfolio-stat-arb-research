"""Reproducible raw/adjusted integrity audit and prospective 400-stock screen.

The screen is diagnostic. It cannot turn current candidates into a historical
security master or certify suspension/borrow/security-lineage evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def audit(source: Path, output: Path, calendar_path: Path | None = None):
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((source / "manifest.json").read_text())
    if manifest["status"] != "downloaded_pending_quality_and_universe_gates":
        raise ValueError("Incomplete collection cannot pass integrity audit")
    candidates = pd.read_csv(source / "equity_candidates.csv")
    calendar = pd.read_csv(calendar_path or source / "market_calendar.csv")
    sessions = pd.DatetimeIndex(pd.to_datetime(calendar.date)).sort_values()
    end = pd.Timestamp(manifest["requested_end_exclusive"])
    sessions = sessions[(sessions >= pd.Timestamp(manifest["requested_start"])) & (sessions < end)]
    checks, frames, files = {}, {}, {}
    for name in ["raw", "all"]:
        path = source / f"bars_{name}.csv"
        checksum = hashlib.sha256(path.read_bytes()).hexdigest()
        if checksum != manifest["files"][path.name]["sha256"]:
            raise ValueError(f"Input checksum mismatch: {path.name}")
        f = pd.read_csv(path)
        f["session"] = pd.to_datetime(f.t, utc=True).dt.tz_convert("America/New_York").dt.tz_localize(None).dt.normalize()
        values = f[["o", "h", "l", "c", "v"]].to_numpy(dtype=float)
        valid = np.isfinite(values).all(axis=1) & (values[:, :4] > 0).all(axis=1) & (values[:, 4] >= 0)
        valid &= (f.h >= f[["o","l","c"]].max(axis=1)).to_numpy()
        valid &= (f.l <= f[["o","h","c"]].min(axis=1)).to_numpy()
        duplicate = f.duplicated(["symbol", "session"], keep=False)
        non_session = ~f.session.isin(sessions)
        bad = f[~valid | duplicate | non_session].copy()
        bad.to_csv(output / f"invalid_{name}_bars.csv", index=False)
        checks[name] = {"rows": len(f), "symbols": int(f.symbol.nunique()),
                        "invalid_ohlcv": int((~valid).sum()), "duplicate_rows": int(duplicate.sum()),
                        "outside_requested_calendar": int(non_session.sum()),
                        "zero_volume": int(f.v.eq(0).sum())}
        files[path.name] = {"sha256": checksum, "bytes": path.stat().st_size}
        # Invalid records remain in their explicit exception file and cannot qualify.
        f["usable"] = valid & ~duplicate & ~non_session & f.v.gt(0).to_numpy()
        frames[name] = f
    raw, adjusted = frames["raw"], frames["all"]
    reconciliation = raw[["symbol","session"]].merge(
        adjusted[["symbol","session"]], on=["symbol","session"], how="outer", indicator=True)
    unmatched = reconciliation[reconciliation._merge.ne("both")]
    unmatched.to_csv(output / "unmatched_raw_adjusted.csv", index=False)
    # One independently supplied market calendar; missing SPY never erases a date.
    rows = []
    close = raw.pivot(index="session", columns="symbol", values="c").reindex(sessions)
    volume = raw.pivot(index="session", columns="symbol", values="v").reindex(sessions)
    usable = raw.pivot(index="session", columns="symbol", values="usable").reindex(sessions).eq(True)
    stale = close.diff().eq(0).rolling(5, min_periods=5).sum().eq(5)
    valid = usable & ~stale
    for row in candidates.itertuples(index=False):
        symbol = row.symbol
        reasons = []
        if symbol not in close:
            rows.append({"symbol":symbol,"security_id":row.security_id,"eligible":False,
                         "reasons":"no_bars","sector":row.sector})
            continue
        history = int(valid[symbol].sum())
        coverage = float(valid[symbol].iloc[-60:].mean())
        mdv = float((close[symbol]*volume[symbol]).where(valid[symbol]).iloc[-60:].median())
        last = float(close[symbol].iloc[-1])
        if not np.isfinite(last) or last < 5:
            reasons.append("price_below_5_or_missing")
        if not valid[symbol].iloc[-1]:
            reasons.append("missing_zero_volume_stale_or_invalid_latest")
        if history < 252:
            reasons.append("history_below_252")
        if coverage < .95:
            reasons.append("coverage_below_95pct")
        if not np.isfinite(mdv) or mdv < 20_000_000:
            reasons.append("sip_median_dollar_volume_below_20m")
        if row.status != "active" or not row.tradable_now:
            reasons.append("inactive_or_untradable_at_capture")
        rows.append({"symbol":symbol, "security_id":row.security_id, "sector":row.sector,
                     "latest_raw_close":last, "median_dollar_volume_60":mdv,
                     "coverage_60":coverage, "valid_observations":history,
                     "shortable_at_capture":row.shortable_now,
                     "eligible":not reasons, "reasons":";".join(reasons)})
    screen = pd.DataFrame(rows)
    selected = screen.loc[screen.eligible].sort_values(
        ["median_dollar_volume_60","security_id"], ascending=[False,True]).head(400).copy()
    selected["membership_available_at"] = manifest["available_at"]
    selected["scope"] = "prospective_diagnostic_only"
    selected["approved_weight"] = 0.0
    screen.to_csv(output / "stock_eligibility_audit.csv", index=False)
    selected.to_csv(output / "prospective_400_stocks.csv", index=False)
    history_coverage = raw.groupby("symbol").agg(first=("session","min"), last=("session","max"),rows=("session","size"))
    history_coverage.to_csv(output / "stock_history_coverage.csv")
    # Large adjusted returns are diagnostics, never winsorized away to hide bad data.
    adjusted_prices = adjusted.pivot(index="session",columns="symbol",values="c").reindex(sessions)
    returns = adjusted_prices.pct_change(fill_method=None)
    extremes = returns.where(returns.abs() > .30).stack().rename("adjusted_return").reset_index()
    extremes.to_csv(output / "large_adjusted_returns.csv", index=False)
    selected.groupby("sector").size().rename("selected_stocks").to_csv(output / "selected_sector_counts.csv")
    result = {"status":"DATA_GATE_NOT_HISTORICAL_PIT", "source":str(source),
              "feed":manifest["feed"], "requested_equity_candidates":len(candidates),
              "eligible_stocks":int(screen.eligible.sum()), "selected_stocks":len(selected),
              "snapshot_available_at":manifest["available_at"], "last_bar_session":str(sessions[-1].date()),
              "calendar_sessions":len(sessions), "checks":checks,
              "unmatched_raw_adjusted_rows":len(unmatched), "large_adjusted_returns":len(extremes),
              "files":files, "approved_orders":0,
              "blocking_gates":["Historical membership publication times unverified",
                                "Delisted security prices and terminal proceeds not reconciled",
                                "Historical identifier lineage and sector vintages unverified",
                                "Suspension events and live borrow checks incomplete",
                                "No accepted walk-forward/OOS evidence or prospective paper history"]}
    (output / "data_audit.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    body = f"""# 股票数据核验（过程记录，非十二周最终结果）

已采集 {len(candidates)} 只股票候选及 SPY。SIP 原始日线 {len(raw):,} 条，复权日线 {len(adjusted):,} 条。
交易日历为 Alpaca 独立日历；没有用 SPY 缺失日期缩减交易日。

按原始收盘价 ≥ $5、有效历史 ≥ 252 个交易日、最近 60 个交易日覆盖 ≥ 95%、
SIP 60 日中位日成交金额 ≥ $20m、最近日有效且非连续不变价格，得到 {int(screen.eligible.sum())} 只合格候选，按流动性排序选出 {len(selected)} 只。
名单见 `prospective_400_stocks.csv`，每只的排除原因见 `stock_eligibility_audit.csv`。

原始/复权键不匹配：{len(unmatched)}；异常复权单日收益绝对值 > 30%：{len(extremes)}，单独列出待核查。

该名单仅是自 {manifest['available_at']} 起可见的当前股票池筛选，不能用于回填 2018–2026 历史成分。
历史持仓、发布时点、证券身份和退市终值尚需核验，因此 approved_weight 均为零，没有订单。
503 只当前股票的行情下载完成不等于历史 PIT 数据层完成。
"""
    (output / "DATA_STATUS.md").write_text(body)
    print(json.dumps({k:result[k] for k in ["status","eligible_stocks","selected_stocks","unmatched_raw_adjusted_rows","large_adjusted_returns"]},indent=2))
    return result


if __name__ == "__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--source",type=Path,default=Path("output/equities_2026-09-20_sip_r2"))
    p.add_argument("--output",type=Path,default=Path("reports/equity_v2/data_quality"))
    p.add_argument("--calendar",type=Path)
    args=p.parse_args()
    audit(args.source,args.output,args.calendar)

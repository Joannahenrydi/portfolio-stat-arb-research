"""Archive a prospective equity candidate universe; NEVER backfill membership.

Only read-only endpoints. Credentials come from environment or two stdin lines.
All observations become available at capture time, including old bar revisions.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

HOLDINGS_URL = "https://www.ishares.com/us/products/239726/ishares-core-s-p-500-etf/latest-holdings.csv"


def utcnow():
    return datetime.now(timezone.utc).isoformat()


class Reader:
    def __init__(self, key, secret, output):
        self.headers = {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}
        self.output = output
        self.requests = []

    def get(self, base, endpoint, params=None):
        url = base + endpoint + ("?" + urlencode(params) if params else "")
        for attempt in range(4):
            try:
                req = Request(url, headers=self.headers)
                with urlopen(req, timeout=60) as r:
                    body = r.read()
                    item = {"endpoint": endpoint, "params": params, "status": r.status,
                            "request_id": r.headers.get("X-Request-ID"),
                            "captured_at": utcnow(), "bytes": len(body),
                            "sha256": hashlib.sha256(body).hexdigest()}
                self.requests.append(item)
                self.flush()
                return json.loads(body)
            except HTTPError as e:
                detail = e.read().decode("utf-8", errors="replace")[:500]
                self.requests.append({"endpoint": endpoint, "status": e.code,
                                      "detail": detail,
                                      "captured_at": utcnow(),
                                      "request_id": e.headers.get("X-Request-ID")})
                self.flush()
                if e.code not in (429, 500, 502, 503, 504) or attempt == 3:
                    raise RuntimeError(f"Alpaca read failed: HTTP {e.code} at {endpoint}: {detail}") from None
                time.sleep(min(20, 2**attempt))
        raise RuntimeError("request exhausted")

    def flush(self):
        (self.output / "requests.json").write_text(json.dumps(self.requests, indent=2))


def collect(output, feed, start, end):
    output.mkdir(parents=True, exist_ok=False)
    observation_start = utcnow()
    key, secret = os.environ.get("APCA_API_KEY_ID"), os.environ.get("APCA_API_SECRET_KEY")
    if not key or not secret:
        raise RuntimeError("Alpaca credentials required in process environment")
    reader = Reader(key, secret, output)
    manifest = {"status": "collecting", "scope": "prospective_only",
                "observation_started_at": observation_start,
                "requested_start": start, "requested_end_exclusive": end,
                "feed": feed, "historical_membership_backfill_allowed": False,
                "historical_backtest_permitted": False, "files": {},
                "limitations": ["Current candidate list is not historical S&P 500 membership.",
                                "Old bars first observed now cannot prove historical availability.",
                                "Provider security ids are not a verified historical lineage map.",
                                "Suspension history and delisting returns not yet reconciled."]}

    def save_manifest():
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2))

    save_manifest()
    # Probe the requested feed explicitly: no silent IEX/SIP substitution.
    reader.get("https://data.alpaca.markets", "/v2/stocks/bars",
               {"symbols": "SPY", "timeframe": "1Day", "feed": feed,
                "start": "2026-09-17", "end": "2026-09-18", "limit": 10})
    with urlopen(Request(HOLDINGS_URL, headers={"User-Agent": "Mozilla/5.0"}), timeout=60) as r:
        body = r.read()
    text = body.decode("utf-8-sig")
    if "Ticker,Name,Sector,Asset Class" not in text:
        raise ValueError("Issuer response is not a verified holdings CSV")
    (output / "issuer_holdings.csv").write_bytes(body)
    rows = list(csv.reader(io.StringIO(text)))
    asof = next(row[1] for row in rows if row and row[0] == "Fund Holdings as of")
    header = next(i for i, row in enumerate(rows) if row[:2] == ["Ticker", "Name"])
    holdings = pd.DataFrame([r for r in rows[header+1:] if len(r) == len(rows[header])],
                            columns=rows[header])
    holdings = holdings.loc[holdings["Asset Class"].eq("Equity")].copy()
    if not 450 <= len(holdings) <= 550:
        raise ValueError("Unexpected equity holdings count")
    manifest["issuer"] = {"source": HOLDINGS_URL, "holdings_asof": asof,
                          "captured_at": utcnow(), "equity_candidates": len(holdings)}
    assets = reader.get("https://paper-api.alpaca.markets", "/v2/assets",
                        {"status": "active", "asset_class": "us_equity"})
    inactive = reader.get("https://paper-api.alpaca.markets", "/v2/assets",
                          {"status": "inactive", "asset_class": "us_equity"})
    (output / "asset_master_snapshot.json").write_text(json.dumps(assets + inactive))
    by_symbol = {a["symbol"]: a for a in assets}
    matched, unresolved = [], []
    for _, row in holdings.iterrows():
        candidates = [row.Ticker, row.Ticker.replace(" ", "."), row.Ticker.replace(" ", "-")]
        matches = [s for s in dict.fromkeys(candidates) if s in by_symbol]
        if len(matches) != 1:
            unresolved.append({"issuer_ticker": row.Ticker, "reason": "unresolved_or_ambiguous"})
            continue
        a = by_symbol[matches[0]]
        matched.append({"symbol": matches[0], "security_id": a["id"], "sector": row.Sector,
                        "name": row.Name, "exchange": a["exchange"],
                        "status": a["status"], "tradable_now": a.get("tradable"),
                        "shortable_now": a.get("shortable"),
                        "easy_to_borrow_now": a.get("easy_to_borrow"),
                        "available_at": utcnow()})
    pd.DataFrame(matched).to_csv(output / "equity_candidates.csv", index=False)
    manifest["mapping_issues"] = unresolved
    manifest["matched_equities"] = len(matched)
    calendar = reader.get("https://paper-api.alpaca.markets", "/v2/calendar",
                          {"start": start, "end": end})
    pd.DataFrame(calendar).to_csv(output / "market_calendar.csv", index=False)
    symbols = sorted(m["symbol"] for m in matched) + ["SPY"]
    for adjustment in ("raw", "all"):
        dest = output / f"bars_{adjustment}.csv"
        with dest.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["symbol","t","o","h","l","c","v","n","vw"])
            writer.writeheader()
            total = 0
            for batch_no, offset in enumerate(range(0, len(symbols), 20)):
                batch = symbols[offset:offset+20]
                params = {"symbols": ",".join(batch), "timeframe": "1Day", "feed": feed,
                          "adjustment": adjustment,
                          "start": pd.Timestamp(start).tz_localize("America/New_York").isoformat(),
                          "end": (pd.Timestamp(end).tz_localize("America/New_York")
                                  - pd.Timedelta(seconds=1)).isoformat(),
                          "asof": "-", "limit": 10000, "sort": "asc"}
                tokens = set()
                while True:
                    result = reader.get("https://data.alpaca.markets", "/v2/stocks/bars", params)
                    if not isinstance(result.get("bars"), dict):
                        raise TypeError("missing bars response")
                    for symbol, bars in result["bars"].items():
                        if symbol not in batch:
                            raise ValueError("unexpected symbol in bars")
                        for bar in bars:
                            writer.writerow({"symbol": symbol, **bar})
                            total += 1
                    token = result.get("next_page_token")
                    if not token:
                        break
                    if token in tokens:
                        raise ValueError("repeated pagination token")
                    tokens.add(token)
                    params["page_token"] = token
                handle.flush()
                print(f"{adjustment}: batch {batch_no+1}, {total:,} bars", flush=True)
        manifest["files"][dest.name] = {"rows": total,
            "sha256": hashlib.sha256(dest.read_bytes()).hexdigest()}
        save_manifest()
    manifest["status"] = "downloaded_pending_quality_and_universe_gates"
    manifest["available_at"] = utcnow()
    manifest["valid_from"] = manifest["available_at"]
    save_manifest()
    print(json.dumps({k: manifest[k] for k in ["status","matched_equities","feed","valid_from"]}))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--feed", choices=["sip","iex"], default="sip")
    p.add_argument("--start", default="2017-01-01")
    p.add_argument("--end", default="2026-09-19")
    p.add_argument("--credentials-stdin", action="store_true")
    a = p.parse_args()
    if a.credentials_stdin:
        os.environ["APCA_API_KEY_ID"] = sys.stdin.readline().strip()
        os.environ["APCA_API_SECRET_KEY"] = sys.stdin.readline().strip()
    collect(a.output, a.feed, a.start, a.end)

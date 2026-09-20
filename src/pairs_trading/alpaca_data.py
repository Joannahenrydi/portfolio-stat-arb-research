"""Explicit-feed, read-only Alpaca bars; no trading endpoint is used."""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd


def download_alpaca(symbols, start, end, output, feed="iex", fetch=None):
    """Download complete daily bars; end is exclusive in America/New_York.

    Raw and adjusted bars are separate research artifacts. They are not compatible
    with the dividend-cash-flow ledger without a corporate-actions reconciliation.
    """
    if feed not in ("iex", "sip"):
        raise ValueError("feed must be iex or sip; no automatic feed fallback")
    if not symbols or len(set(symbols)) != len(symbols):
        raise ValueError("distinct symbols required")
    begin = pd.Timestamp(start).tz_localize("America/New_York")
    finish = pd.Timestamp(end).tz_localize("America/New_York")
    if begin >= finish or finish > pd.Timestamp.now(tz="America/New_York").normalize():
        raise ValueError("end is exclusive and must not exceed today's New York midnight")
    if fetch is None:
        key, secret = os.getenv("APCA_API_KEY_ID"), os.getenv("APCA_API_SECRET_KEY")
        if not key or not secret:
            raise RuntimeError("Configure APCA_API_KEY_ID and APCA_API_SECRET_KEY locally")

        def fetch(params):
            request = Request("https://data.alpaca.markets/v2/stocks/bars?" + urlencode(params),
                              headers={"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret})
            with urlopen(request, timeout=30) as response:
                return json.load(response)

    path = Path(output)
    path.mkdir(parents=True, exist_ok=False)
    metadata = {"provider": "Alpaca", "feed": feed, "symbols": symbols,
                "start": start, "end_exclusive": end, "status": "incomplete",
                "retrieved_at_utc": datetime.now(timezone.utc).isoformat(), "files": {},
                "limitations": ["IEX is one exchange, not consolidated market prices.",
                                "No forward filling; market-calendar completeness not verified.",
                                "Adjusted bars are research proxies, not executable prices.",
                                "Corporate actions required before cash-flow ledger use."]}
    manifest = path / "metadata.json"
    manifest.write_text(json.dumps(metadata, indent=2) + "\n")
    for adjustment in ("raw", "all"):
        params = {"symbols": ",".join(symbols), "timeframe": "1Day", "feed": feed,
                  "adjustment": adjustment, "start": begin.isoformat(),
                  "end": (finish-timedelta(seconds=1)).isoformat(),
                  "asof": "-", "limit": 10000, "sort": "asc"}
        rows, tokens = [], set()
        while True:
            result = fetch(dict(params))
            if not isinstance(result.get("bars"), dict):
                raise TypeError("missing Alpaca bars; check data entitlement")
            for symbol, bars in result["bars"].items():
                if symbol not in symbols:
                    raise ValueError("unexpected response symbol")
                rows.extend({"symbol": symbol, **bar} for bar in bars)
            token = result.get("next_page_token")
            if not token:
                break
            if token in tokens:
                raise ValueError("repeated pagination token")
            tokens.add(token)
            params["page_token"] = token
        frame = pd.DataFrame(rows)
        required = {"symbol", "t", "o", "h", "l", "c", "v"}
        if not required.issubset(frame.columns) or set(frame.symbol) != set(symbols):
            raise ValueError("missing symbols or bar fields")
        frame["t"] = pd.to_datetime(frame.t, utc=True)
        values = frame[["o", "h", "l", "c", "v"]].to_numpy(dtype=float)
        if (frame.duplicated(["symbol", "t"]).any() or not np.isfinite(values).all()
                or (values[:, :4] <= 0).any() or (values[:, 4] < 0).any()
                or (frame.h < frame[["o", "l", "c"]].max(axis=1)).any()
                or (frame.l > frame[["o", "h", "c"]].min(axis=1)).any()
                or (frame.t < begin).any() or (frame.t >= finish).any()):
            raise ValueError("invalid, duplicate or out-of-range bars")
        frame = frame.sort_values(["symbol", "t"])
        target = path / f"bars_{adjustment}.csv"
        frame.to_csv(target, index=False)
        metadata["files"][target.name] = {"rows": len(frame),
            "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
            "rows_by_symbol": frame.groupby("symbol").size().to_dict()}
        manifest.write_text(json.dumps(metadata, indent=2) + "\n")
    metadata["status"] = "downloaded_not_execution_validated"
    manifest.write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata

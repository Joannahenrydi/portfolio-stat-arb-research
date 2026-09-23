"""Read-only auxiliary archive: exchange calendar and corporate actions.

This preserves provider responses and process dates, not invented publication
times. Corporate-action completeness and historical lineage remain audit gates.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import pandas as pd
from collect_equity_snapshot import Reader, utcnow


def collect(source: Path, output: Path, symbols_file: Path | None = None):
    output.mkdir(parents=True, exist_ok=False)
    keys = [os.environ.get("APCA_API_KEY_ID"), os.environ.get("APCA_API_SECRET_KEY")]
    if not all(keys):
        raise RuntimeError("Missing process-local Alpaca credentials")
    reader = Reader(*keys, output)
    source_meta = json.loads((source / "manifest.json").read_text())
    symbols = pd.read_csv(source / "equity_candidates.csv").symbol.tolist()
    if symbols_file:
        extra = pd.read_csv(symbols_file)
        symbols += extra["ticker"].dropna().astype(str).tolist()
    symbols = sorted(set(symbols))
    calendar = reader.get("https://paper-api.alpaca.markets", "/v2/calendar",
                          {"start": "2016-12-01", "end": "2026-12-31"})
    pd.DataFrame(calendar).to_csv(output / "market_calendar.csv", index=False)
    manifest = {"status": "collecting", "captured_started_at": utcnow(),
                "source_snapshot": str(source), "symbols": symbols,
                "historical_published_at_verified": False,
                "request_source_feed": source_meta["feed"], "pages": []}
    index = output / "manifest.json"
    index.write_text(json.dumps(manifest, indent=2))
    total = 0
    for batch_no, begin in enumerate(range(0, len(symbols), 40)):
        params = {"symbols": ",".join(symbols[begin:begin+40]),
                  "start": "2017-01-01", "end": "2026-09-21",
                  "limit": 1000, "data_quality": "all", "sort": "asc"}
        seen = set()
        page = 0
        while True:
            result = reader.get("https://data.alpaca.markets", "/v1/corporate-actions", params)
            if "corporate_actions" not in result:
                raise ValueError("Corporate actions response missing required field")
            dest = output / f"actions_{batch_no:03}_{page:03}.json"
            content = json.dumps(result, sort_keys=True).encode()
            dest.write_bytes(content)
            grouped = result["corporate_actions"]
            count = sum(len(v) for v in grouped.values()) if isinstance(grouped, dict) else len(grouped)
            total += count
            manifest["pages"].append({"file": dest.name, "actions": count,
                                      "captured_at": utcnow(),
                                      "sha256": hashlib.sha256(content).hexdigest()})
            index.write_text(json.dumps(manifest, indent=2))
            token = result.get("next_page_token")
            if not token:
                break
            if token in seen:
                raise ValueError("Repeated corporate-action page token")
            seen.add(token)
            params["page_token"] = token
            page += 1
        print(f"Corporate actions batch {batch_no+1}: cumulative {total:,}", flush=True)
    manifest.update(status="downloaded_pending_reconciliation", actions=total,
                    captured_at=utcnow())
    index.write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, default=Path("output/equities_2026-09-20_sip_r2"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--extra-symbols", type=Path)
    p.add_argument("--credentials-stdin", action="store_true")
    args = p.parse_args()
    if args.credentials_stdin:
        os.environ["APCA_API_KEY_ID"] = sys.stdin.readline().strip()
        os.environ["APCA_API_SECRET_KEY"] = sys.stdin.readline().strip()
    collect(args.source, args.output, args.extra_symbols)

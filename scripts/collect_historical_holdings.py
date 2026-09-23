"""Archive dated IVV holdings without inventing historical publication times.

Read-only public issuer requests. Raw responses are immutable, downloads resume
from hash-checked snapshots, and returned as-of dates must exactly match requests.
CUSIP/ISIN are preserved reference identifiers, not asserted permanent lineage.
This is a historical candidate source, not an executable/PIT-certified universe.
"""
from __future__ import annotations

import argparse
import calendar
import csv
import hashlib
import io
import json
import re
import sys
import time
from collections import Counter, defaultdict
from datetime import date, datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

ENDPOINT = (
    "https://www.ishares.com/varnish-api/blk-one01-product-data/"
    "product-data/api/v2/get-product-data"
)
FIELDS = {
    "ticker": "ticker", "issueName": "name", "assetClass": "asset_class",
    "sectorName": "sector", "cusip": "cusip", "isin": "isin", "sedol": "sedol",
    "countryOfRisk": "country_of_risk", "exchange": "exchange",
    "holdingPercent": "weight_percent", "unitPrice": "issuer_valuation_price",
    "unitsHeld": "shares_held", "marketValue": "market_value",
    "currencyCode": "currency",
}
ROW_FIELDS = [
    "requested_asof", "returned_asof", "reference_id", *FIELDS.values(),
    "captured_at", "source_url", "raw_sha256", "published_at", "available_at",
    "publication_time_confidence", "permanent_lineage_verified",
]
AUDIT_FIELDS = [
    "requested_asof", "returned_asof", "status", "calendar_basis", "rows",
    "equity_rows", "equity_missing_isin", "equity_missing_cusip",
    "duplicate_equity_reference_ids", "captured_at", "raw_sha256", "raw_file",
    "published_at", "available_at", "publication_time_confidence", "error",
]
SENTINELS = {"", "-", "--", "N/A", "None"}


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def digest(body: bytes) -> str:
    return hashlib.sha256(body).hexdigest()


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def atomic_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def schedule(start: date, end: date, calendar_path: Path) -> tuple[list[dict], dict]:
    """Use an explicit session calendar; first boundary may be supplied separately."""
    if end < start:
        raise ValueError("end must follow start")
    body = calendar_path.read_bytes()
    calendar_rows = list(csv.DictReader(io.StringIO(body.decode("utf-8-sig"))))
    if not calendar_rows:
        raise ValueError("empty session calendar")
    field = "date" if "date" in calendar_rows[0] else "session"
    sessions = [date.fromisoformat(row[field]) for row in calendar_rows]
    if sessions != sorted(set(sessions)):
        raise ValueError("session calendar must be ordered and unique")
    by_month: dict[str, list[date]] = defaultdict(list)
    for session in sessions:
        by_month[session.strftime("%Y-%m")].append(session)
    plan = []
    year, month = start.year, start.month
    while (year, month) <= (end.year, end.month):
        key = f"{year:04}-{month:02}"
        month_end = date(year, month, calendar.monthrange(year, month)[1])
        if key in by_month:
            requested = max(by_month[key])
            if sessions[-1] < month_end:
                raise ValueError(f"calendar does not establish the full month {key}")
            basis = "supplied_session_calendar"
        elif (year, month) == (start.year, start.month) and start < sessions[0]:
            requested = start
            basis = "explicit_start_boundary_not_calendar_verified"
        else:
            raise ValueError(f"calendar has no sessions for {key}")
        if start <= requested <= end:
            plan.append({"requested_asof": requested.isoformat(), "calendar_basis": basis})
        year, month = (year + 1, 1) if month == 12 else (year, month + 1)
    if not plan or plan[0]["requested_asof"] != start.isoformat():
        raise ValueError("start must be the initial monthly snapshot date")
    if plan[-1]["requested_asof"] != end.isoformat():
        raise ValueError("end must be the final monthly snapshot date")
    return plan, {
        "path": str(calendar_path.resolve()), "sha256": digest(body),
        "rows": len(sessions), "first_session": sessions[0].isoformat(),
        "last_session": sessions[-1].isoformat(),
        "role": "Date scheduling only; no claim of contemporaneous calendar capture.",
    }


def source_url(requested_asof: str) -> str:
    return ENDPOINT + "?" + urlencode({
        "appSubType": "ISHARES", "appType": "PRODUCT_PAGE", "component": "holdings.all",
        "locale": "en_US", "portfolioId": "239726", "targetSite": "us-ishares",
        "userType": "individual", "excludeContent": "true",
        "asOfDate": requested_asof.replace("-", ""), "includeConfig": "true",
    })


def clean_identifier(value) -> str:
    text = str(value).strip() if value is not None else ""
    return "" if text in SENTINELS else text


def observed_asof(body: bytes) -> str | None:
    """Report the source date even for responses rejected by exact-date validation."""
    try:
        points = json.loads(body)["componentsByNameMap"]["holdings"]["containersByNameMap"][
            "all"
        ]["dataPointsByNameMap"]
        value = str(points["asOfDate"]["value"])
        if re.fullmatch(r"\d{8}", value):
            return date(int(value[:4]), int(value[4:6]), int(value[6:8])).isoformat()
    except (ValueError, KeyError, TypeError):
        pass
    return None


def parse_snapshot(body: bytes, requested_asof: str) -> tuple[list[dict], dict]:
    try:
        data = json.loads(body)
        points = data["componentsByNameMap"]["holdings"]["containersByNameMap"]["all"][
            "dataPointsByNameMap"
        ]
        returned = str(points["asOfDate"]["value"])
    except (ValueError, KeyError, TypeError) as exc:
        raise ValueError("response is not the required issuer holdings JSON schema") from exc
    if not re.fullmatch(r"\d{8}", returned):
        raise ValueError(f"invalid internal asOfDate: {returned!r}")
    returned_asof = date(int(returned[:4]), int(returned[4:6]), int(returned[6:8])).isoformat()
    if returned_asof != requested_asof:
        raise ValueError(
            f"DATE_MISMATCH: requested {requested_asof}, returned {returned_asof}; "
            "no fallback/current substitution accepted"
        )
    columns = {}
    for source, target in FIELDS.items():
        values = points.get(source, {}).get("value")
        if not isinstance(values, list):
            raise ValueError(f"missing/non-array holdings column {source}")  # noqa: TRY004
        columns[target] = values
    lengths = {len(values) for values in columns.values()}
    if len(lengths) != 1:
        raise ValueError("holdings columns have unequal lengths")
    rows = []
    for index in range(next(iter(lengths))):
        row = {key: values[index] for key, values in columns.items()}
        for key in ("isin", "cusip", "sedol", "ticker"):
            row[key] = clean_identifier(row[key])
        row["reference_id"] = (
            "ISIN:" + row["isin"] if row["isin"] else
            "CUSIP:" + row["cusip"] if row["cusip"] else ""
        )
        row.update(requested_asof=requested_asof, returned_asof=returned_asof)
        rows.append(row)
    equities = [row for row in rows if row["asset_class"] == "Equity"]
    if not 450 <= len(equities) <= 550:
        raise ValueError(f"unexpected IVV equity count: {len(equities)}")
    keys = Counter(row["reference_id"] for row in equities if row["reference_id"])
    summary = {
        "returned_asof": returned_asof, "rows": len(rows), "equity_rows": len(equities),
        "equity_missing_isin": sum(not row["isin"] for row in equities),
        "equity_missing_cusip": sum(not row["cusip"] for row in equities),
        "duplicate_equity_reference_ids": sum(count - 1 for count in keys.values()),
    }
    return rows, summary


def cached_snapshot(output: Path, item: dict) -> dict | None:
    path = output / "snapshots" / f"{item['requested_asof']}.json"
    if not path.exists():
        return None
    meta = json.loads(path.read_text())
    body = (output / meta["raw_file"]).read_bytes()
    if digest(body) != meta["raw_sha256"]:
        raise ValueError(f"cached raw hash mismatch: {path}")
    _, checked = parse_snapshot(body, item["requested_asof"])
    if meta["requested_asof"] != item["requested_asof"]:
        raise ValueError(f"cached requested date mismatch: {path}")
    for key, value in checked.items():
        if meta.get(key) != value:
            raise ValueError(f"cached metadata differs from raw data: {path}, {key}")
    return meta


def fetch_snapshot(output: Path, item: dict, timeout: float, retries: int) -> dict:
    url = source_url(item["requested_asof"])
    for attempt in range(retries + 1):
        started = utcnow()
        status, body, headers, error = None, b"", {}, ""
        try:
            req = Request(url, headers={
                "Accept": "application/json", "x-application-id": "pp-ui-csr",
                "User-Agent": "historical-holdings-research/1.0",
            })
            with urlopen(req, timeout=timeout) as response:
                body = response.read()
                status = response.status
                headers = {key: response.headers.get(key) for key in (
                    "Content-Type", "Date", "Last-Modified", "ETag", "Age",
                )}
        except HTTPError as exc:
            status, body, error = exc.code, exc.read(), str(exc)
        except (URLError, TimeoutError, OSError) as exc:
            error = f"{type(exc).__name__}: {exc}"
        captured = utcnow()
        sha = digest(body)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        raw_path = output / "raw" / item["requested_asof"] / f"{stamp}_{sha[:16]}.json"
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        # Exclusive creation preserves each response, including rejected/error bodies.
        with raw_path.open("xb") as handle:
            handle.write(body)
        meta = {
            **item, "source_url": url, "request_started_at": started,
            "captured_at": captured, "http_status": status, "http_headers": headers,
            "raw_file": str(raw_path.relative_to(output)), "raw_sha256": sha,
            "bytes": len(body), "attempt": attempt + 1, "published_at": None,
            "available_at": None, "publication_time_confidence": "unverified_historical_release",
            "historical_publication_verified": False, "permanent_lineage_verified": False,
            "returned_asof": observed_asof(body), "error": error,
        }
        if status == 200:
            try:
                _, summary = parse_snapshot(body, item["requested_asof"])
                meta.update(summary, status="accepted_historical_snapshot")
            except ValueError as exc:
                meta.update(status="rejected_response", error=str(exc))
        else:
            meta["status"] = "request_failed"
        atomic_json(raw_path.with_suffix(".metadata.json"), meta)
        with (output / "requests.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(meta, ensure_ascii=False) + "\n")
        if meta["status"] == "accepted_historical_snapshot":
            atomic_json(output / "snapshots" / f"{item['requested_asof']}.json", meta)
            return meta
        if status == 200 or status not in (None, 429, 500, 502, 503, 504) or attempt == retries:
            return meta
        time.sleep(min(8, 2 ** (attempt + 1)))
    raise RuntimeError("unreachable retry state")


def export_audit(output: Path, plan: list[dict], calendar_meta: dict, failures: dict) -> dict:
    journal = output / "requests.jsonl"
    attempts = [json.loads(line) for line in journal.read_text().splitlines()] if journal.exists() else []
    latest_failures = {}
    for attempt in attempts:
        if attempt["status"] != "accepted_historical_snapshot":
            latest_failures[attempt["requested_asof"]] = attempt
    latest_failures.update(failures)
    failures = latest_failures
    audit, holdings, accepted = [], [], []
    for item in plan:
        try:
            meta = cached_snapshot(output, item)
        except (ValueError, KeyError, OSError) as exc:
            meta = None
            failures[item["requested_asof"]] = {**item, "status": "cache_rejected", "error": str(exc)}
        if meta is None:
            rejected = failures.get(item["requested_asof"], {**item, "status": "pending"}).copy()
            if rejected.get("raw_file"):
                rejected["returned_asof"] = observed_asof((output / rejected["raw_file"]).read_bytes())
            audit.append(rejected)
            continue
        accepted.append(meta)
        audit.append(meta)
        rows, _ = parse_snapshot((output / meta["raw_file"]).read_bytes(), item["requested_asof"])
        for row in rows:
            row.update({key: meta[key] for key in (
                "captured_at", "source_url", "raw_sha256", "published_at", "available_at",
                "publication_time_confidence", "permanent_lineage_verified",
            )})
        holdings.extend(rows)
    atomic_csv(output / "snapshot_audit.csv", AUDIT_FIELDS, audit)
    atomic_csv(output / "holdings.csv", ROW_FIELDS, holdings)
    equities = [row for row in holdings if row["asset_class"] == "Equity"]
    union: dict[str, list[dict]] = defaultdict(list)
    for row in equities:
        if row["reference_id"]:
            union[row["reference_id"]].append(row)
    union_rows = []
    last_date = max((meta["returned_asof"] for meta in accepted), default=None)
    for key, rows in sorted(union.items()):
        union_rows.append({
            "reference_id": key, "isin": "|".join(sorted({r["isin"] for r in rows})),
            "cusips_observed": "|".join(sorted({r["cusip"] for r in rows})),
            "tickers_observed": "|".join(sorted({r["ticker"] for r in rows})),
            "names_observed": "|".join(sorted({r["name"] for r in rows})),
            "sectors_observed": "|".join(sorted({r["sector"] for r in rows})),
            "first_snapshot": min(r["returned_asof"] for r in rows),
            "last_snapshot": max(r["returned_asof"] for r in rows),
            "snapshot_count": len({r["returned_asof"] for r in rows}),
            "present_in_latest_snapshot": any(r["returned_asof"] == last_date for r in rows),
            "delisting_status": "not_inferred_from_holdings_absence",
            "permanent_lineage_verified": False,
        })
    union_fields = [
        "reference_id", "isin", "cusips_observed", "tickers_observed", "names_observed",
        "sectors_observed", "first_snapshot", "last_snapshot", "snapshot_count",
        "present_in_latest_snapshot", "delisting_status", "permanent_lineage_verified",
    ]
    atomic_csv(output / "equity_union.csv", union_fields, union_rows)
    counts = [meta["equity_rows"] for meta in accepted]
    manifest = {
        "schema_version": 1, "source": "iShares IVV official historical holdings JSON",
        "endpoint": ENDPOINT, "rebuilt_at": utcnow(), "calendar": calendar_meta,
        "requested_start": plan[0]["requested_asof"], "requested_end": plan[-1]["requested_asof"],
        "requested_snapshots": len(plan), "accepted_snapshots": len(accepted),
        "pending_or_failed_snapshots": len(plan) - len(accepted),
        "status": "collection_complete" if len(accepted) == len(plan) else "collection_incomplete",
        "all_accepted_dates_match_requests": all(
            row["requested_asof"] == row["returned_asof"] for row in accepted
        ),
        "rejected_date_mismatch_requests": sum(
            "DATE_MISMATCH" in attempt.get("error", "") for attempt in attempts
        ),
        "unavailable_snapshot_dates": [
            row["requested_asof"] for row in audit if row["status"] != "accepted_historical_snapshot"
        ],
        "holdings_rows": len(holdings), "equity_rows": len(equities),
        "equity_count_min": min(counts, default=0), "equity_count_max": max(counts, default=0),
        "unique_equity_reference_ids": len(union),
        "unique_equity_ticker_labels": len({row["ticker"] for row in equities}),
        "reference_ids_absent_latest": sum(not r["present_in_latest_snapshot"] for r in union_rows),
        "historical_pit_gate": "BLOCKED_PENDING_PUBLICATION_AND_LINEAGE_EVIDENCE",
        "historical_backtest_permitted": False,
        "limitations": [
            "Issuer holdings snapshots are not certified historical S&P 500 index membership.",
            "Historical as-of date is verified; original publication/availability is not.",
            "Capture and HTTP headers refer to this retrieval, not original publication.",
            "ISIN/CUSIP can change at corporate actions; no permanent lineage is fabricated.",
            "Holdings absence is not a delisting; all observed equity reference IDs are retained.",
            "Monthly snapshots miss intramonth entries/exits and are only a candidate policy input.",
            "Sector values preserve issuer responses; historical classification vintage needs audit.",
            "Issuer valuation prices are not OHLCV, consolidated volume, or executable quotes.",
            "Delisted OHLCV, terminal consideration, suspensions, and borrow history remain separate.",
        ],
        "files": {name: {"sha256": digest((output / name).read_bytes())} for name in (
            "snapshot_audit.csv", "holdings.csv", "equity_union.csv",
        )},
    }
    atomic_json(output / "manifest.json", manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=date.fromisoformat, default=date(2016, 12, 30))
    parser.add_argument("--end", type=date.fromisoformat, default=date(2026, 8, 31))
    parser.add_argument("--calendar", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("output/historical_ivv_holdings"))
    parser.add_argument("--timeout", type=float, default=30)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--pause", type=float, default=0.5)
    parser.add_argument("--force", action="store_true", help="capture new versions; preserve old raw")
    parser.add_argument("--audit-only", action="store_true", help="rebuild local tables; no network")
    args = parser.parse_args()
    if args.timeout <= 0 or args.retries < 0 or args.pause < 0:
        parser.error("timeout must be positive; retries/pause must be nonnegative")
    plan, calendar_meta = schedule(args.start, args.end, args.calendar)
    args.output.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output / "requested_snapshots.json", plan)
    failures = {}
    for number, item in enumerate(plan, start=1):
        try:
            cached = cached_snapshot(args.output, item)
        except (ValueError, KeyError, OSError) as exc:
            # Preserve corrupt files for investigation; explicit --force is required to replace index.
            if not args.force:
                failures[item["requested_asof"]] = {
                    **item, "status": "cache_rejected", "error": str(exc),
                }
                print(f"{number}/{len(plan)} {item['requested_asof']} CACHE_REJECTED: {exc}", flush=True)
                continue
            cached = None
        if args.audit_only or (cached and not args.force):
            continue
        meta = fetch_snapshot(args.output, item, args.timeout, args.retries)
        if meta["status"] != "accepted_historical_snapshot":
            failures[item["requested_asof"]] = meta
        print(
            f"{number}/{len(plan)} {item['requested_asof']} {meta['status']} "
            f"equities={meta.get('equity_rows', 0)} {meta.get('error', '')}", flush=True,
        )
        if number % 12 == 0:
            export_audit(args.output, plan, calendar_meta, failures)
        time.sleep(args.pause)
    manifest = export_audit(args.output, plan, calendar_meta, failures)
    print(json.dumps({key: manifest[key] for key in (
        "status", "requested_snapshots", "accepted_snapshots", "equity_rows",
        "unique_equity_reference_ids", "historical_pit_gate",
    )}), flush=True)
    return 0 if manifest["pending_or_failed_snapshots"] == 0 else 2


if __name__ == "__main__":
    sys.exit(main())

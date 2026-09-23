# Historical IVV holdings: source evidence and remaining limits

This optional dataset improves the historical candidate list. It does not change the
user's later permission to continue strategy research with disclosed survivorship bias.
The collection's historical-PIT certification gate is separate from the permission to
run that research. No data or source files have been uploaded.

## Verified official source

The [IVV product page](https://www.ishares.com/us/products/239726/ishares-core-sp-500-etf)
loads holdings through its public product-data JSON interface. The collector uses:

```text
https://www.ishares.com/varnish-api/blk-one01-product-data/product-data/api/v2/get-product-data
```

Its parameters are `appSubType=ISHARES`, `appType=PRODUCT_PAGE`,
`component=holdings.all`, `locale=en_US`, `portfolioId=239726`,
`targetSite=us-ishares`, `userType=individual`, `excludeContent=true`,
`includeConfig=true`, and `asOfDate=YYYYMMDD`. Headers identify JSON and the public
page client (`Accept: application/json`, `x-application-id: pp-ui-csr`).

The arrays are under
`componentsByNameMap.holdings.containersByNameMap.all.dataPointsByNameMap`.
Each column's `value` array contains the original source values. Every accepted
response must have an internal `asOfDate.value` exactly equal to the request and
consistent column lengths. Date mismatches, HTML, malformed JSON, missing columns,
and implausible IVV equity counts are rejected. A response's HTTP 200 status or
`Content-Type` alone is insufficient.

This check is material: the old `1467271812596.ajax` URL returned HTML, while
`latest-holdings.csv?asOfDate=20181231` returned September 2026 holdings in a real
CSV. The JSON interface also returned current holdings for several unavailable
2017 dates. The collector preserves these responses and rejects them; it never
substitutes today's holdings or interpolates missing membership.

## Local collection and reproduction

Run from the repository root with a Python 3.10+ environment:

```bash
python scripts/collect_historical_holdings.py \
  --calendar output/equities_2026-09-20_sip_r2/market_calendar.csv \
  --output output/historical_ivv_holdings

# Offline integrity check and table rebuild; does not issue network requests.
python scripts/collect_historical_holdings.py \
  --calendar output/equities_2026-09-20_sip_r2/market_calendar.csv \
  --output output/historical_ivv_holdings --audit-only
```

The local environment used for this collection is `../.venv/bin/python`. Defaults
request 117 monthly snapshots from 2016-12-30 through 2026-08-31. Month-ends from
2017 onward come from the existing, explicitly supplied Alpaca session calendar,
whose bytes and SHA256 are recorded. December 30, 2016 is an explicit requested
initial boundary outside that calendar's coverage and is labelled accordingly.
Neither weekday arithmetic nor the holdings response is presented as an official
exchange calendar. Changing the date span requires both boundary dates to match
the monthly schedule.

All data are under the existing gitignored `output/` directory:

| Path within `output/historical_ivv_holdings/` | Contents |
| --- | --- |
| `raw/<requested-date>/` | Original response bytes and separate retrieval metadata, including rejected responses |
| `snapshots/<date>.json` | Index to an accepted immutable raw response |
| `requests.jsonl` | Every request's date, URL, status, capture time, response hash, and validation outcome |
| `requested_snapshots.json` | Explicit monthly schedule and date provenance |
| `snapshot_audit.csv` | Requested versus returned dates, counts, missing identifiers, and errors |
| `holdings.csv` | All asset classes; identifiers, names, sectors, weights, valuations, provenance |
| `equity_union.csv` | Every observed equity reference identity across accepted snapshots, including former holdings |
| `manifest.json` | Verified counts, completeness, file hashes, and limitations |

A normal rerun verifies hashes and schemas before reusing successful snapshots and
retries unsuccessful dates. `--force` captures a new version but retains older raw
files. The raw response hash is repeated in normalized rows, making their source
recoverable. Derived tables may be regenerated; they are not the immutable archive.

Collection counts and unavailable dates are reported in the final run results below.

## Historical presence, identity, and publication time are distinct

The official archive demonstrably returns old holdings: preliminary verified dates
include 2017-12-29, 2018-01-02, 2018-01-03, 2018-12-31, and 2023-02-28. The 2018
snapshots contain CELG, RTN, ATVI, and CERN. The February 2023 snapshot contains
FRC and SIVB, with issuer valuation prices of $123.01 and $288.11 respectively.
These observations disprove a simple current-constituent backfill for those dates.
They do not certify every archive field as its original historical vintage.

CUSIP and ISIN are stored as strings with leading zeroes intact. An `ISIN:` key
(or `CUSIP:` fallback) groups observations of the same reference identifier.
Neither identifier is asserted to be permanent across reorganizations. The union
retains ticker/name/sector changes, and `permanent_lineage_verified` remains false.
It must be reconciled to a permanent security master before joining price histories
across corporate actions. Removal from IVV is not itself evidence of delisting.
The union therefore records absence from the final snapshot without assigning a
delisting event or deleting the name.

`captured_at` records the actual retrieval time. `published_at` and `available_at`
remain empty, and historical publication confidence is explicitly unverified.
Today's HTTP Date/Last-Modified headers do not prove original historical release.
Conversely, downloading an old document today does not prevent its use in a
historical study when independent archival evidence establishes that the same
information was public earlier. The required evidence is information availability,
not ownership of a contemporaneous personal download.

## Concrete route to a publication-bounded universe

An original public filing can define a deliberately delayed candidate policy:
after a filing became public, consider its disclosed common-stock holdings until
the next public filing, then apply price/liquidity/history filters using information
available at each decision. That is a public-holdings candidate universe, not a
claim to reconstruct exact daily S&P 500 membership.

One verified original filing is [SEC accession 0001752724-25-043800](https://www.sec.gov/Archives/edgar/data/1100663/0001752724-25-043800-index.htm).
The SEC index identifies IVV, registrant CIK `0001100663`, series `S000004310`, and
class `C000012040`; report period 2024-12-31; filing date 2025-02-27; and displayed
acceptance time 2025-02-27 12:18:22. It links the original 513,012-byte NPORT XML
and a holdings exhibit. This establishes a real later knowledge bound for the
original public filing. The date must not be backdated to December 31. Before
using an intraday timestamp, verify the SEC timestamp convention and public
dissemination timing; a next-session decision gives a conservative operational
buffer but is a declared policy, not an invented actual release time.

For the earlier period, an [original BlackRock announcement on the ASX archive](https://announcements.asx.com.au/asxpdf/20180830/pdf/43xw5fsbpbxxpn.pdf)
is dated 2018-08-30 and includes the June 30, 2018 Form N-Q holdings for IVV.
It proves that the attached report was public by that announcement, with a visible
delay from the holdings date. Its first page identifies both dates; IVV's schedule
starts in the early pages of the attachment. A date-only announcement does not
establish an intraday release time. Use a documented later decision boundary.

The [SEC N-PORT bulk archive](https://www.sec.gov/data-research/sec-markets-data/form-n-port-data-sets)
begins in 2019 Q4 and contains only publicly disseminated filings. Original N-Q
and N-CSR reports are needed for earlier periods. No bulk SEC archive was downloaded
for this collection. Filing metadata must be joined by IVV's series, not merely
the trust CIK, which covers many funds.

To certify this policy, obtain each original filing plus its public-release evidence;
extract holdings from the original document; reconcile share classes; compare the
same report-period list to iShares JSON; and retain discrepancies and amendments
as separately dated versions. Prefer the original filing's list where the lists
differ. A filing publication time cannot simply be attached to an unverified later
JSON revision. The [issuer's portfolio disclosure policy](https://www.ishares.com/us/literature/sai/sai-ishares-inc-eo-8-31.pdf)
describes daily disclosure before the market opens for the covered funds, but this
current document alone neither establishes IVV's full historical policy nor proves
that each archived response is the originally disclosed version. Blanket
next-business-day publication timestamps are therefore not manufactured.

## Limits on using the collection

Monthly holdings are candidates, not daily listing/suspension states. They miss
intramonth index events and require an explicit lagged candidate policy. Issuer
valuation prices are not consolidated OHLCV or executable quotes. The collection
does not supply historical borrow availability, terminal merger consideration,
delisting returns, or historical exchange/suspension events. Source sector labels
are preserved, but original classification vintage still requires reconciliation.
No failed date, missing price, or missing identifier is silently filled.

The collector's integrity checks cover exact-date rejection, invalid HTML, unequal
column lengths, immutable raw hashes, and resume validation. The script passes
the repository's Ruff rules. All results remain local.

## Final run results

Pending completion of the currently running download; `manifest.json` is the
machine-readable source of truth and is refreshed during collection.

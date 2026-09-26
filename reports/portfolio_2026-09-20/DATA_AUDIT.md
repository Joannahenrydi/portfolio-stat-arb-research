# Data Collection and Coverage Audit

This run downloaded and authenticated 35,322 raw Alpaca IEX daily bars for 23 symbols plus
ALL-adjusted bars over the same range. It also collected 55,776 Yahoo bars as the primary research
data specified by the frozen protocol.

Alpaca requested 2017–2026, but most symbols begin on 2020-07-27 and SPY has one isolated earlier
record. Missing history was not fabricated and the short series is not represented as a complete
2018–2022 training set. IEX is a single-venue feed; liquidity thresholds and capacity estimates are
feed proxies rather than total-market capacity.

Among adjacent same-date/same-symbol observations available from both sources, 897 returns differ
by more than one percentage point. See `source_return_discrepancies.csv`. Differences include feed
closes, missing bars and adjustment factors; neither source was automatically rewritten. These
cross-source discrepancies prevent treating performance from one source as directly executable on
the other.

Raw files are `output/portfolio_v1/alpaca_data/bars_raw.csv` and `bars_all.csv`; Yahoo inputs and
metadata are in `output/portfolio_v1/market_data`. `data_file_hashes.json` records every SHA256, and
`source_coverage.csv` records per-symbol start/end dates. Credentials were never written to project
files; the process environment supplied them only for read-only downloads, and no order endpoint
was called.

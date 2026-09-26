# Equity Data Audit (Interim Record, Not the Final 12-Week Result)

Collected 503 equity candidates plus SPY: 1,161,734 raw SIP daily bars and 1,161,734 adjusted daily
bars. The trading calendar is an independent Alpaca calendar; missing SPY dates never remove market
sessions.

The screen requires raw close >= $5, at least 252 valid sessions, at least 95% coverage over the
latest 60 sessions, SIP 60-session median dollar volume >= $20 million, a valid latest bar, and no
stale-price run. It produced 499 eligible candidates and selected 400 by liquidity. See
`prospective_400_stocks.csv`; per-name exclusion reasons are in `stock_eligibility_audit.csv`.

Unmatched raw/adjusted keys: 0. Adjusted daily returns with absolute value above 30%: 122, retained
separately for review.

This list is a current-universe screen available only from 2026-09-20T07:00:27.670424+00:00; it
cannot backfill 2018–2026 membership. Historical holdings, publication timestamps, security identity
and delisting terminal values remain unverified, so every approved_weight is zero and there are no
orders. Downloading current history for 503 stocks does not complete a historical PIT data layer.

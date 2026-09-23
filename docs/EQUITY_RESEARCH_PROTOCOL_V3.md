# Equity research protocol v3

Frozen on 2026-09-23 after the v2 audit was observed, before v3 candidate results.
Consequently the 2025–2026 segment is reused evidence, not a pristine holdout. The next
uncontaminated out-of-sample evidence must be accumulated prospectively.

The data scope, continuity screen, causal close-to-next-close timing, neutralization,
$100,000 capacity basis, 1% name cap, 25% turnover cap, 10 bps ADV participation cap,
and cost model are unchanged from `EQUITY_RESEARCH_PROTOCOL.md`.

## Finite development grid

- Alpha input: raw stock return or market-and-sector residual return.
- Lookback: 63 sessions skipping 5, 126 skipping 21, or 252 skipping 21.
- Portfolio score: continuous cross-sectional rank or the top/bottom 20% or 30% tails.
- Rebalance interval: 5, 10, or 21 market sessions.

This gives 54 candidates. Select using only 2018–2022 training and 2023–2024 validation.
A candidate must have positive net Sharpe in both segments, validation drawdown no worse
than 25%, average absolute drifted net below 1%, and annual turnover below 80 times NAV.
Rank by the smaller of training and validation Sharpe, then their mean, then validation
CAGR, then name. Report every candidate. If none pass, v3 is rejected.

The chosen v3 candidate may be run on 2025–2026 only as reused robustness evidence. It
cannot restore a fresh holdout, guarantee returns, or authorize orders.

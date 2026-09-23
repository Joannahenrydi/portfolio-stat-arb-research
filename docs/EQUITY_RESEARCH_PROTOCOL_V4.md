# Equity research protocol v4

Frozen on 2026-09-23 after v3 was rejected and before v4 candidate results. The
2018–2022/2023–2024 development split and all data, continuity, causality, exposure,
capacity, and cost rules remain unchanged. The 2025–2026 period has already been seen in
v2 and can only be reused as secondary evidence.

## Finite alpha grid

The following signals are computed at close `t` and first earn return `t+1`:

- negative 60- and 126-session residual volatility;
- 252-session price position relative to its trailing high;
- 63- and 126-session residual trend divided by residual volatility;
- 63- and 126-session residual path efficiency (sum divided by sum of absolute moves);
- 21- and 63-session cumulative adjusted overnight gap;
- 21- and 63-session cumulative adjusted intraday return;
- equal-rank blend of 252-session price position and negative 60-session residual
  volatility.

Each signal uses either a continuous rank or top/bottom 20% tails and rebalances every
10 or 21 market sessions: 48 candidates total. A candidate must have positive net Sharpe
in both development segments, validation drawdown no worse than 25%, validation average
absolute drifted net below 1%, and annual turnover below 80 times NAV. Rank by the smaller
development Sharpe, their mean, validation CAGR, then name. If none pass, v4 is rejected.

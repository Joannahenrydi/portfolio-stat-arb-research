# v11 Train-Qualified OHLCV Alpha Discovery

## Decision

**REJECTED.** The frozen train-only filter admitted `reversal_5`, `intraday_reversal`, and
`volume_shock_reversal`. None produced stable positive net performance across train and the reused
2023-2024 development interval. The 2025-2026 audit was not loaded and orders remain disabled.

The identity audit removed 13 symbols with at least 20 internal missing/zero-volume sessions. This
includes the FISV discontinuity that invalidated several v10 candidates. The correction is based on
bar continuity, not returns, but it adds complete-history bias and does not upgrade the snapshot to
PIT evidence.

## Train-only feature qualification

| Feature | Mean rank IC | ICIR | Positive train years | Result |
|---|---:|---:|---:|---|
| reversal_5 | 0.00894 | 0.0652 | 4/5 | admitted |
| intraday_reversal | 0.00868 | 0.0651 | 5/5 | admitted |
| volume_shock_reversal | 0.00402 | 0.0355 | 4/5 | admitted |
| momentum_126_21 | 0.01205 | 0.0906 | 3/5 | rejected: unstable by year |
| reversal_1 | 0.00533 | 0.0448 | 4/5 | rejected: correlated |
| momentum_252_21 | -0.01674 | -0.1451 | 1/5 | rejected |

The full qualification table is in `train_feature_qualification.csv`.

## Portfolio result

| Candidate | Construction | Train net Sharpe | Development net Sharpe | Development net CAGR |
|---|---|---:|---:|---:|
| A00 | v10 reference | 0.134 | -0.350 | -0.24% |
| A01 | reversal_5, 10-day | -0.433 | **0.023** | **0.02%** |
| A02 | top-two equal | -0.293 | -1.087 | -0.87% |
| A03 | three-family ICIR | -0.238 | -0.896 | -0.72% |
| A06 | reversal ensemble, 5-day | 0.120 | -0.964 | -0.78% |
| A07 | three-family, 2x gross | -0.443 | -1.770 | -2.45% |

Train-positive IC did not survive portfolio costs and regime change. Leverage amplified a negative
net edge and therefore cannot produce acceptance. The remaining path is independent information,
not more OHLCV feature or rebalance search.

## Next research boundary

v12 should add filing-date fundamental information from SEC EDGAR. Company Facts and Frames carry
filing dates and require no API key, allowing a causal profitability/growth/accrual family. Any SEC
candidate must retain the same costs, neutrality, train/development gates, and prospective-only
promotion requirement.


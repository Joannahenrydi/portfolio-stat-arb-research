from __future__ import annotations

from itertools import combinations

import pandas as pd


def is_i1(series: pd.Series, significance: float = 0.05) -> bool:
    """Require levels to look non-stationary and first differences stationary."""
    from statsmodels.tsa.stattools import adfuller

    clean = series.dropna().astype(float)
    return adfuller(clean, autolag="AIC")[1] > significance and adfuller(
        clean.diff().dropna(), autolag="AIC"
    )[1] < significance


def johansen_pair(x: pd.Series, y: pd.Series, significance_index: int = 1) -> bool:
    """Johansen trace rank >= 1. significance_index: 0=90%, 1=95%, 2=99%."""
    from statsmodels.tsa.vector_ar.vecm import coint_johansen

    frame = pd.concat([x, y], axis=1).dropna()
    result = coint_johansen(frame, det_order=0, k_ar_diff=1)
    return bool(result.lr1[0] > result.cvt[0, significance_index])


def select_pairs(prices: pd.DataFrame, require_i1: bool = True) -> list[tuple[str, str]]:
    selected = []
    for x, y in combinations(prices.columns, 2):
        if require_i1 and not (is_i1(prices[x]) and is_i1(prices[y])):
            continue
        if johansen_pair(prices[x], prices[y]):
            selected.append((x, y))
    return selected


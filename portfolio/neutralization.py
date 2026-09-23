"""Build and audit portfolio exposure constraints."""

from __future__ import annotations

import numpy as np
import pandas as pd


def build_exposure_matrix(
    beta: pd.Series,
    sectors: pd.Series,
    factors: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Return labeled beta, sector and optional style exposures for one session."""
    if not beta.index.is_unique or not sectors.index.is_unique or set(beta.index) != set(sectors.index):
        raise ValueError("beta and sectors must share unique security IDs")
    index = beta.index
    if beta.isna().any() or sectors.isna().any() or not np.isfinite(beta).all():
        raise ValueError("beta and sector exposures must be complete")
    result = pd.concat(
        [beta.astype(float).rename("market_beta"), pd.get_dummies(sectors, prefix="sector", dtype=float)],
        axis=1,
    )
    if factors is not None:
        if not factors.index.is_unique or set(factors.index) != set(index):
            raise ValueError("factor exposures must share the security IDs")
        factors = factors.reindex(index).astype(float)
        if not np.isfinite(factors).all().all():
            raise ValueError("factor exposures must be finite")
        overlap = set(result.columns) & set(factors.columns)
        if overlap:
            raise ValueError(f"duplicate exposure names: {sorted(overlap)}")
        result = pd.concat([result, factors], axis=1)
    return result.reindex(index)


def exposure_report(weights: pd.Series, exposure: pd.DataFrame) -> pd.Series:
    """Report dollar, beta, sector and factor exposure of a target portfolio."""
    if not weights.index.is_unique or not exposure.index.is_unique:
        raise ValueError("weights and exposures require unique security IDs")
    if set(weights.index) != set(exposure.index):
        raise ValueError("weights and exposures must share security IDs")
    values = weights.reindex(exposure.index).astype(float)
    matrix = exposure.astype(float)
    if not np.isfinite(values).all() or not np.isfinite(matrix).all().all():
        raise ValueError("weights and exposures must be finite")
    return pd.concat([pd.Series({"dollar_net": float(values.sum())}), matrix.T @ values])

import numpy as np
import pandas as pd
import pytest

from backtest.conditional import (
    ConditionalConfig,
    StrictNeutralityError,
    fit_conditioning_artifact,
    regime_gross_scalar,
    strict_neutral_target,
)


def inputs(n=60):
    names = pd.Index([f"s{i}" for i in range(n)])
    score = pd.Series(np.linspace(-1, 1, n), index=names)
    beta = pd.Series(np.tile([0.8, 1.0, 1.2], n // 3), index=names)
    sectors = pd.Series(np.repeat(["a", "b", "c"], n // 3), index=names)
    previous = pd.Series(0.0, index=names)
    adv = pd.Series(20_000_000.0, index=names)
    residual_volatility = pd.Series(np.linspace(0.01, 0.04, n), index=names)
    return score, beta, sectors, previous, adv, residual_volatility


def test_strict_target_is_dollar_beta_and_sector_neutral_after_drift_correction():
    args = list(inputs())
    args[3].iloc[:3] = [0.003, -0.002, 0.001]
    target, metrics = strict_neutral_target(*args, 1.0, ConditionalConfig())
    assert abs(target.sum()) < 1e-8
    assert abs((target * args[1]).sum()) < 1e-8
    assert target.groupby(args[2]).sum().abs().max() < 1e-8
    assert metrics["max_linear_exposure"] < 1e-8
    assert metrics["turnover"] <= 0.25


def test_infeasible_neutrality_correction_rejects_instead_of_partial_target():
    args = list(inputs())
    args[3].iloc[:2] = [0.01, -0.005]
    args[4][:] = 1.0
    with pytest.raises(StrictNeutralityError, match="LIQUIDITY"):
        strict_neutral_target(*args, 1.0, ConditionalConfig())


def test_residual_volatility_conditioning_keeps_full_universe():
    base = strict_neutral_target(*inputs(), 1.0, ConditionalConfig())[0]
    conditional = strict_neutral_target(
        *inputs(), 1.0, ConditionalConfig(residual_vol_mode="mild")
    )[0]
    assert (base != 0).sum() == len(base)
    assert (conditional != 0).sum() == len(conditional)
    assert not np.allclose(base, conditional)


def test_regime_scaler_uses_frozen_training_distribution_only():
    dates = pd.bdate_range("2020-01-01", periods=600)
    dispersion = pd.Series(np.linspace(0.01, 0.03, len(dates)), index=dates)
    artifact = fit_conditioning_artifact(dispersion, dates[0], dates[399])
    before = regime_gross_scalar("mild", 0.02, -0.10, artifact)
    changed = dispersion.copy()
    changed.loc[dates[400]:] = 100.0
    replay = fit_conditioning_artifact(changed, dates[0], dates[399])
    after = regime_gross_scalar("mild", 0.02, -0.10, replay)
    assert before == after
    assert 0.35 <= before <= 1.0

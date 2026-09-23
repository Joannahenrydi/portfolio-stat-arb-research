import numpy as np
import pandas as pd
import pytest

from backtest.residual_reversal import (
    ResidualReversalConfig,
    ResidualReversalError,
    build_v7_scores,
    fit_stress_artifact,
    stress_gross_scalar,
    strict_liquid_target,
)


def test_v7_core_score_matches_frozen_formula():
    index = pd.date_range("2020-01-01", periods=6, freq="B")
    residual = pd.DataFrame(
        {
            "A": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "B": [2.0, 1.0, 4.0, 3.0, 6.0, 5.0],
            "C": [3.0, 4.0, 1.0, 2.0, 1.0, 2.0],
        },
        index=index,
    )
    eligible = residual.notna()
    scores = build_v7_scores(residual, eligible)
    one = residual.sub(residual.mean(axis=1), axis=0).div(residual.std(axis=1), axis=0)
    raw3 = residual.rolling(3).sum()
    z3 = raw3.sub(raw3.mean(axis=1), axis=0).div(raw3.std(axis=1), axis=0)
    raw5 = residual.rolling(5).sum()
    z5 = raw5.sub(raw5.mean(axis=1), axis=0).div(raw5.std(axis=1), axis=0)
    expected = -(0.5 * one + 0.3 * z3 + 0.2 * z5)
    pd.testing.assert_frame_equal(scores["core"], expected)
    pd.testing.assert_frame_equal(scores["reference"], -z3)


def test_stress_scaler_is_continuous_and_missing_is_invalid():
    index = pd.date_range("2018-01-01", periods=300, freq="B")
    series = pd.Series(np.arange(1, 301, dtype=float), index=index)
    artifact = fit_stress_artifact(series, index[0], index[-1])
    assert stress_gross_scalar(100, artifact) == pytest.approx(1.0)
    assert stress_gross_scalar(285, artifact) == pytest.approx(0.5)
    assert 0.25 < stress_gross_scalar(296, artifact) < 0.5
    assert stress_gross_scalar(400, artifact) == pytest.approx(0.25)
    with pytest.raises(ResidualReversalError, match="MISSING_STRESS_INPUT"):
        stress_gross_scalar(np.nan, artifact)


def test_dynamic_eligibility_liquidates_old_name_and_remains_neutral():
    names = pd.Index([f"S{i:02d}" for i in range(60)])
    score = pd.Series(np.sin(np.arange(len(names)) * 0.7), index=names)
    beta = pd.Series(np.linspace(0.7, 1.3, len(names)), index=names)
    sectors = pd.Series([f"G{i % 3}" for i in range(len(names))], index=names)
    previous = pd.Series(0.0, index=names)
    previous.iloc[0] = 0.001
    previous.iloc[1] = -0.001
    adv = pd.Series(100_000_000.0, index=names)
    residual_vol = pd.Series(np.linspace(0.01, 0.04, len(names)), index=names)
    eligibility = pd.Series(True, index=names)
    eligibility.iloc[0] = False
    target, metrics = strict_liquid_target(
        score,
        beta,
        sectors,
        previous,
        adv,
        residual_vol,
        eligibility,
        1.0,
        ResidualReversalConfig(),
    )
    assert target.iloc[0] == 0
    assert abs(target.sum()) < 1e-8
    assert abs((target * beta).sum()) < 1e-8
    assert target.groupby(sectors).sum().abs().max() < 1e-8
    assert metrics["turnover"] <= 0.25 + 1e-8

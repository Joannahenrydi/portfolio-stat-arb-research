import numpy as np
import pandas as pd

from features.adaptive_alpha import (
    AdaptiveKalmanConfig,
    adaptive_kalman_innovation,
    apply_confidence_filter,
    cost_hurdle,
    expected_return,
    fit_zero_intercept_calibration,
    orthogonalize_families,
)


def test_adaptive_kalman_uses_prior_q_and_r():
    rng = np.random.default_rng(7)
    index = pd.bdate_range("2020-01-01", periods=150)
    market = pd.Series(rng.normal(0, 0.01, len(index)), index=index)
    returns = pd.DataFrame(
        {"A": market.to_numpy() + rng.normal(0, 0.005, len(index))}, index=index
    )
    config = AdaptiveKalmanConfig(warmup=20)
    original = adaptive_kalman_innovation(returns, market, config=config)
    altered_returns = returns.copy()
    altered_returns.loc[index[100], "A"] += 0.20
    altered = adaptive_kalman_innovation(altered_returns, market, config=config)

    pd.testing.assert_series_equal(
        original.q_prior.loc[: index[100], "A"], altered.q_prior.loc[: index[100], "A"]
    )
    pd.testing.assert_series_equal(
        original.r_prior.loc[: index[100], "A"], altered.r_prior.loc[: index[100], "A"]
    )
    assert altered.r_prior.loc[index[101], "A"] > original.r_prior.loc[index[101], "A"]
    assert altered.signal.loc[index[100], "A"] != original.signal.loc[index[100], "A"]


def test_daily_gram_schmidt_removes_cross_family_exposure():
    rng = np.random.default_rng(11)
    index = pd.bdate_range("2022-01-03", periods=4)
    columns = [f"S{i:03d}" for i in range(100)]
    first = pd.DataFrame(rng.normal(size=(4, 100)), index=index, columns=columns)
    second = 0.8 * first + pd.DataFrame(
        rng.normal(scale=0.3, size=(4, 100)), index=index, columns=columns
    )
    eligibility = pd.DataFrame(True, index=index, columns=columns)
    orthogonal = orthogonalize_families(
        {"first": first, "second": second}, eligibility, minimum_names=50
    )
    for session in index:
        correlation = orthogonal["first"].loc[session].corr(orthogonal["second"].loc[session])
        assert abs(correlation) < 1e-12


def test_cost_filter_requires_expected_edge_and_complete_inputs():
    index = pd.bdate_range("2020-01-01", periods=400)
    columns = [f"S{i:03d}" for i in range(300)]
    rng = np.random.default_rng(17)
    score = pd.DataFrame(rng.normal(size=(400, 300)), index=index, columns=columns)
    labels = score * 0.002 + pd.DataFrame(
        rng.normal(scale=0.001, size=score.shape), index=index, columns=columns
    )
    label_end = pd.DataFrame(
        np.tile(index.to_numpy()[:, None], (1, len(columns))), index=index, columns=columns
    )
    artifact = fit_zero_intercept_calibration(
        score, labels, label_end, index[0], index[-1]
    )
    assert artifact.slope > 0
    forecast = expected_return(score, artifact)
    volatility = pd.DataFrame(0.02, index=index, columns=columns)
    adv = pd.DataFrame(50_000_000.0, index=index, columns=columns)
    hurdle = cost_hurdle(forecast, volatility, adv)
    filtered, passed = apply_confidence_filter(score, forecast, hurdle, 1.0)
    assert filtered.notna().equals(passed)
    assert passed.to_numpy().any()
    assert (~passed).to_numpy().any()

    adv.iloc[0, 0] = np.nan
    missing_hurdle = cost_hurdle(forecast, volatility, adv)
    _, missing_passed = apply_confidence_filter(score, forecast, missing_hurdle, 1.0)
    assert not missing_passed.iloc[0, 0]

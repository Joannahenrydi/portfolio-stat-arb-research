from dataclasses import fields

import numpy as np
import pandas as pd
import pytest

from pairs_trading.equity_alphas import (
    AlphaConfig,
    AlphaDataError,
    build_equity_features,
    estimate_shrinkage_covariance,
    fit_expected_return_calibration,
    forward_total_return_labels,
    predict_expected_returns,
)


def panel():
    rng = np.random.default_rng(942)
    dates = pd.bdate_range("2020-01-01", periods=240)
    ids = [f"permanent-{n}" for n in range(6)]
    market = pd.Series(rng.normal(0, 0.01, len(dates)), index=dates)
    noise = rng.normal(0, 0.006, (len(dates), len(ids)))
    sector = np.repeat(rng.normal(0, 0.003, (len(dates), 2)), 3, axis=1)
    returns = pd.DataFrame(
        market.to_numpy()[:, None] * np.linspace(0.7, 1.3, 6) + sector + noise,
        index=dates,
        columns=ids,
    )
    volume = pd.DataFrame(rng.uniform(1e6, 3e6, returns.shape), index=dates, columns=ids)
    sectors = pd.DataFrame([["A"] * 3 + ["B"] * 3] * len(dates), index=dates, columns=ids)
    eligibility = pd.DataFrame(True, index=dates, columns=ids)
    config = AlphaConfig(
        market_window=20,
        sector_window=20,
        volatility_window=15,
        volume_window=15,
        kalman_min_observations=10,
    )
    return returns, market, volume, sectors, eligibility, config


def fitted_fixture():
    returns, market, volume, sectors, eligible, config = panel()
    features = build_equity_features(returns, market, volume, sectors, eligible, config)
    labels, ends = forward_total_return_labels(returns, 3)
    artifact = fit_expected_return_calibration(
        features.signals,
        labels,
        ends,
        train_start=returns.index[50],
        train_end=returns.index[180],
        pit_verified=True,
        min_samples=500,
    )
    return returns, features, labels, ends, artifact


def test_every_feature_is_prefix_invariant_and_future_membership_is_not_used():
    returns, market, volume, sectors, eligible, config = panel()
    cutoff = returns.index[160]
    whole = build_equity_features(returns, market, volume, sectors, eligible, config)
    short = build_equity_features(
        returns.loc[:cutoff],
        market.loc[:cutoff],
        volume.loc[:cutoff],
        sectors.loc[:cutoff],
        eligible.loc[:cutoff],
        config,
    )
    for field in fields(whole):
        if field.name == "signals":
            for family in whole.signals:
                pd.testing.assert_frame_equal(
                    whole.signals[family].loc[:cutoff], short.signals[family]
                )
        else:
            pd.testing.assert_frame_equal(
                getattr(whole, field.name).loc[:cutoff], getattr(short, field.name)
            )
    sectors.loc[sectors.index > cutoff] = "new-sector"
    eligible.loc[eligible.index > cutoff] = False
    altered = build_equity_features(returns, market, volume * 1.0, sectors, eligible, config)
    for name in whole.signals:
        pd.testing.assert_frame_equal(
            whole.signals[name].loc[:cutoff], altered.signals[name].loc[:cutoff]
        )


def test_current_shock_cannot_enter_current_rolling_parameters_or_kalman_prior():
    returns, market, volume, sectors, eligible, config = panel()
    at = returns.index[120]
    before = build_equity_features(returns, market, volume, sectors, eligible, config)
    shocked = returns.copy()
    shocked.loc[at, "permanent-0"] += 0.2
    after = build_equity_features(shocked, market, volume, sectors, eligible, config)
    for field in (
        "market_beta_prior",
        "sector_beta_prior",
        "kalman_beta_prior",
        "volatility_prior",
    ):
        pd.testing.assert_series_equal(
            getattr(before, field).loc[at], getattr(after, field).loc[at]
        )
    innovation_change = (
        after.kalman_innovation.loc[at, "permanent-0"]
        - before.kalman_innovation.loc[at, "permanent-0"]
    )
    assert innovation_change == pytest.approx(0.2)
    assert after.kalman_beta_prior.iloc[121, 0] != before.kalman_beta_prior.iloc[121, 0]


def test_missing_returns_are_not_filled_and_sector_factors_exclude_self():
    returns, market, volume, sectors, eligible, config = panel()
    at = returns.index[120]
    before = build_equity_features(returns, market, volume, sectors, eligible, config)
    expected = before.market_residual.loc[at, ["permanent-1", "permanent-2"]].mean()
    assert before.sector_factor.loc[at, "permanent-0"] == pytest.approx(expected)
    returns.loc[at, "permanent-0"] = np.nan
    after = build_equity_features(returns, market, volume, sectors, eligible, config)
    for frame in after.signals.values():
        assert pd.isna(frame.loc[at, "permanent-0"])
    assert pd.isna(after.kalman_innovation.loc[at, "permanent-0"])
    assert pd.isna(after.market_beta_prior.iloc[121, 0])


def test_universe_ineligible_names_never_receive_signals():
    returns, market, volume, sectors, eligible, config = panel()
    eligible.iloc[-10:, 0] = False
    result = build_equity_features(returns, market, volume, sectors, eligible, config)
    assert all(frame.iloc[-10:, 0].isna().all() for frame in result.signals.values())


def test_forward_labels_include_only_future_sessions_and_propagate_missing_returns():
    idx = pd.date_range("2022-01-01", periods=5)
    returns = pd.DataFrame({"permanent-0": [0.9, 0.1, 0.2, np.nan, 0.4]}, index=idx)
    labels, end = forward_total_return_labels(returns, 2)
    assert labels.iloc[0, 0] == pytest.approx(1.1 * 1.2 - 1)
    assert end.iloc[0, 0] == idx[2]
    assert labels.iloc[1:, 0].isna().all()
    assert end.iloc[-2:, 0].isna().all()


def test_calibration_uses_only_complete_labels_inside_training_segment():
    returns, features, labels, ends, artifact = fitted_fixture()
    assert artifact.sample_count == (180 - 50 + 1 - 3) * 6
    assert artifact.latest_label_session == artifact.train_end
    altered_signals = {name: frame.copy() for name, frame in features.signals.items()}
    outside = (returns.index < artifact.train_start) | (returns.index > artifact.train_end)
    for frame in altered_signals.values():
        frame.loc[outside] = 1e6
    changed_labels = labels.copy()
    changed_labels.loc[outside] = -1e8
    changed_labels.loc[ends.iloc[:, 0] > artifact.train_end] = 1e8
    fitted = fit_expected_return_calibration(
        altered_signals,
        changed_labels,
        ends,
        train_start=artifact.train_start,
        train_end=artifact.train_end,
        pit_verified=True,
        min_samples=500,
    )
    assert artifact == fitted


def test_training_artifact_is_frozen_and_prediction_cannot_run_in_training_segment():
    returns, features, _labels, _ends, artifact = fitted_fixture()
    heldout = {
        name: frame.loc[returns.index > artifact.train_end]
        for name, frame in features.signals.items()
    }
    predictions = predict_expected_returns(heldout, artifact)
    assert predictions.notna().all().all()
    first = next(iter(heldout.values())).index[0]
    expected = artifact.intercept
    for j, family in enumerate(artifact.families):
        x = heldout[family].loc[first, "permanent-0"]
        expected += (
            (x - artifact.feature_means[j]) / artifact.feature_scales[j] * artifact.coefficients[j]
        )
    assert predictions.loc[first, "permanent-0"] == pytest.approx(expected)
    with pytest.raises(AlphaDataError, match="strictly after"):
        predict_expected_returns(features.signals, artifact)
    heldout[artifact.families[0]].iloc[0, 0] = np.nan
    assert pd.isna(predict_expected_returns(heldout, artifact).iloc[0, 0])


def test_pit_training_is_required_unless_snapshot_research_is_explicit():
    returns, features, labels, ends, artifact = fitted_fixture()
    kwargs = {
        "train_start": artifact.train_start,
        "train_end": artifact.train_end,
        "min_samples": 500,
    }
    with pytest.raises(AlphaDataError, match="verified PIT"):
        fit_expected_return_calibration(
            features.signals, labels, ends, pit_verified=False, **kwargs
        )
    research = fit_expected_return_calibration(
        features.signals,
        labels,
        ends,
        pit_verified=False,
        input_scope="research_snapshot_only",
        **kwargs,
    )
    assert not research.pit_verified
    assert research.input_scope == "research_snapshot_only"
    heldout = {
        name: frame.loc[returns.index > artifact.train_end]
        for name, frame in features.signals.items()
    }
    assert predict_expected_returns(heldout, research).notna().all().all()
    with pytest.raises(AlphaDataError, match="pit_verified=False"):
        fit_expected_return_calibration(
            features.signals,
            labels,
            ends,
            pit_verified=True,
            input_scope="research_snapshot_only",
            **kwargs,
        )


def test_covariance_excludes_decision_and_future_returns_and_is_positive_definite():
    returns, *_ = panel()
    at = returns.index[180]
    covariance = estimate_shrinkage_covariance(returns, at, lookback=120, min_observations=90)
    future_changed = returns.copy()
    future_changed.loc[future_changed.index >= at] = 100
    changed = estimate_shrinkage_covariance(future_changed, at, lookback=120, min_observations=90)
    pd.testing.assert_frame_equal(covariance.covariance, changed.covariance)
    assert covariance.window_end == returns.index[179]
    assert covariance.observations.eq(120).all()
    assert np.linalg.eigvalsh(covariance.covariance).min() > 0
    expected = returns.iloc[60:180].cov()
    target = np.diag(np.diag(expected))
    np.testing.assert_allclose(covariance.covariance, 0.8 * expected + 0.2 * target)


def test_covariance_missing_values_use_pairwise_counts_and_insufficient_data_gates():
    returns, *_ = panel()
    at = returns.index[180]
    returns.iloc[70:80, 0] = np.nan
    estimate = estimate_shrinkage_covariance(returns, at, lookback=120, min_observations=90)
    assert estimate.observations.iloc[0] == 110
    assert estimate.pairwise_observations.iloc[0, 1] == 110
    returns.iloc[60:150, 0] = np.nan
    with pytest.raises(AlphaDataError, match="pairwise prior"):
        estimate_shrinkage_covariance(returns, at, lookback=120, min_observations=90)


def test_covariance_repairs_pairwise_indefinite_matrix_and_reports_repair():
    idx = pd.bdate_range("2022-01-01", periods=7)
    # Each observed pair is individually coherent but their joint matrix is not.
    values = [
        [1, 1, np.nan],
        [-1, -1, np.nan],
        [1, np.nan, 1],
        [-1, np.nan, -1],
        [np.nan, 1, -1],
        [np.nan, -1, 1],
        [0, 0, 0],
    ]
    returns = pd.DataFrame(values, index=idx, columns=["a", "b", "c"]) / 100
    estimate = estimate_shrinkage_covariance(
        returns,
        idx[-1],
        lookback=6,
        min_observations=2,
        shrinkage=0,
    )
    assert estimate.repaired_eigenvalues >= 1
    assert np.linalg.eigvalsh(estimate.covariance).min() > 0


def test_alpha_inputs_reject_ambiguous_session_alignment():
    returns, market, volume, sectors, eligible, config = panel()
    with pytest.raises(AlphaDataError, match="same sessions"):
        build_equity_features(returns, market, volume.iloc[1:], sectors, eligible, config)
    bad = eligible.copy().astype(object)
    bad.iloc[0, 0] = "False"
    with pytest.raises(AlphaDataError, match="explicit booleans"):
        build_equity_features(returns, market, volume, sectors, bad, config)

import numpy as np
import pandas as pd
import pytest

from portfolio.optimizer import (
    PortfolioConstraints,
    PortfolioCosts,
    construct_portfolio,
    estimate_costs,
)


def inputs(n=8):
    names = pd.Index([f"security-{i}" for i in range(n)])
    alpha = pd.Series(np.tile([0.005, -0.005], n // 2), index=names)
    covariance = pd.DataFrame(np.eye(n) * 0.0004, index=names, columns=names)
    # Deliberately redundant: net equals the sum of the two sector columns.
    exposure = pd.DataFrame({"net": 1.0, "beta": np.tile([0.8, 0.8, 1.2, 1.2], n // 4),
                             "sector_a": np.arange(n) < n // 2,
                             "sector_b": np.arange(n) >= n // 2}, index=names)
    previous = pd.Series(0.0, index=names)
    adv = pd.Series(10_000_000.0, index=names)
    return alpha, covariance, exposure, previous, adv, 100_000.0


def test_neutral_feasible_target_reserves_costs_and_preserves_labels():
    args = inputs()
    result = construct_portfolio(*args)
    assert result.status == "ACCEPTED", result.metrics
    w, metrics, status = result
    assert status == "ACCEPTED"
    assert w.index.equals(args[0].index)
    assert abs(w.sum()) < 1e-7
    assert np.abs(args[2].T @ w).max() < 1e-7
    assert metrics["turnover"] <= 0.25 + 1e-7
    assert metrics["max_participation"] <= 0.001 + 1e-7
    assert metrics["post_cost_gross"] <= 1 + 1e-7
    assert metrics["post_cost_max_name"] <= 0.01 + 1e-7
    assert metrics["projected_costs"]["total"] <= metrics["conservative_cost_reserve"] + 1e-7
    assert metrics["assumed_fills"] == 0


def test_infeasible_liquidity_preserves_holdings_instead_of_assumed_liquidation():
    args = list(inputs())
    args[3].iloc[0] = 0.04  # Above the name cap and non-neutral.
    args[4].iloc[0] = 0.0  # No admissible exit order.
    result = construct_portfolio(*args)
    assert result.status == "BLOCKED"
    np.testing.assert_array_equal(result.target, args[3])
    assert result.metrics["turnover"] == 0
    assert result.metrics["assumed_fills"] == 0
    assert "single_name" in result.metrics["current_constraint_violations"]


def test_small_liquidity_budget_is_binding():
    args = list(inputs())
    args[4][:] = 100_000.0
    result = construct_portfolio(*args)
    assert result.status == "ACCEPTED", result.metrics
    assert np.abs(result.target).max() <= 0.001 + 1e-7
    assert result.metrics["max_participation"] <= 0.001 + 1e-7


def test_zero_alpha_rejected_without_forced_trades():
    args = list(inputs())
    args[0][:] = 0.0
    args[3].iloc[:2] = [0.02, -0.02]
    result = construct_portfolio(*args)
    assert result.status == "REJECTED"
    assert result.metrics["reason"] == "NO_ALPHA"
    np.testing.assert_array_equal(result.target, args[3])
    assert result.metrics["execution_allowed"] is False


def test_costs_can_remove_an_unprofitable_forecast():
    args = list(inputs())
    args[0] *= 0.001
    result = construct_portfolio(*args)
    assert result.status == "REJECTED", result.metrics
    assert np.abs(result.target).sum() == 0


def test_exact_cost_is_monotone_and_borrow_uses_calendar_days():
    previous = np.zeros(2)
    target = np.array([0.01, -0.01])
    vol, adv, nav = np.full(2, 0.02), np.full(2, 1e6), 1e5
    base = estimate_costs(previous, target, vol, adv, nav)
    doubled = estimate_costs(previous, target, vol, adv, nav, PortfolioCosts(multiplier=2))
    weekend = estimate_costs(previous, target, vol, adv, nav, PortfolioCosts(calendar_days=3))
    larger = estimate_costs(previous, 2 * target, vol, adv, nav)
    illiquid = estimate_costs(previous, target, vol, adv / 10, nav)
    assert doubled["total"] == pytest.approx(2 * base["total"])
    assert weekend["borrow"] == pytest.approx(3 * base["borrow"])
    assert weekend["transaction"] == pytest.approx(base["transaction"])
    assert larger["impact"] == pytest.approx(base["impact"] * 2 ** 1.5)
    assert illiquid["total"] > base["total"]
    with pytest.raises(ValueError, match="zero ADV"):
        estimate_costs(previous, target, vol, np.zeros(2), nav)


def test_inputs_align_by_security_id_and_covariance_must_be_psd():
    args = list(inputs())
    expected = construct_portfolio(*args)
    args[1] = args[1].iloc[::-1, ::-1]
    args[2] = args[2].iloc[::-1]
    args[3] = args[3].iloc[::-1]
    args[4] = args[4].iloc[::-1]
    actual = construct_portfolio(*args)
    np.testing.assert_allclose(actual.target, expected.target, atol=1e-9)
    args[1].iloc[0, 0] = -1.0
    with pytest.raises(ValueError, match="semidefinite"):
        construct_portfolio(*args)


def test_nonfinite_forecast_and_liquidity_fail_closed():
    args = list(inputs())
    args[0].iloc[0] = np.nan
    result = construct_portfolio(*args)
    assert result.status == "BLOCKED"
    assert result.metrics["reason"] == "NONFINITE_ALPHA"
    args[0].iloc[0] = 0.005
    args[4].iloc[0] = np.nan
    result = construct_portfolio(*args)
    assert result.status == "BLOCKED"
    assert result.metrics["reason"] == "MISSING_OR_INVALID_LIQUIDITY"


def test_cash_only_constraint_cannot_bypass_turnover_to_exit():
    args = list(inputs())
    args[3].iloc[:2] = [0.01, -0.01]
    result = construct_portfolio(*args, constraints=PortfolioConstraints(max_gross=0, max_turnover=0.001))
    assert result.status == "BLOCKED"
    np.testing.assert_array_equal(result.target, args[3])

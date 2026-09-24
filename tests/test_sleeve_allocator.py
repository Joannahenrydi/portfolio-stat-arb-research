import numpy as np
import pandas as pd
import pytest

from portfolio.sleeve_allocator import (
    SleeveAllocationError,
    SleeveAllocatorConfig,
    run_sleeve_allocation,
    solve_sleeve_allocation,
)


def synthetic_fit(periods=126):
    index = pd.date_range("2020-01-01", periods=periods, freq="B")
    market = pd.Series(np.sin(np.arange(periods) / 7) * 0.01, index=index)
    sleeves = pd.DataFrame(
        {
            "market_like": market.to_numpy() + 0.0002,
            "market_opposite": -0.8 * market.to_numpy() + 0.0002,
            "independent": np.cos(np.arange(periods) / 5) * 0.003 + 0.0002,
            "quiet": np.sin(np.arange(periods) / 13) * 0.001 + 0.0002,
            "diverse": np.cos(np.arange(periods) / 11) * 0.002 + 0.0002,
        },
        index=index,
    )
    return sleeves, market


def test_paper_objective_reduces_absolute_market_correlation():
    sleeves, market = synthetic_fit()
    previous = pd.Series(0.2, index=sleeves.columns)
    target, metrics = solve_sleeve_allocation(
        sleeves,
        market,
        previous,
        SleeveAllocatorConfig(return_floor="zero"),
    )
    equal_corr = abs((sleeves.mean(axis=1)).corr(market))
    assert metrics["absolute_correlation"] < equal_corr
    assert target.sum() == pytest.approx(1)
    assert target.max() <= 0.60 + 1e-7
    assert (target - previous).abs().sum() <= 0.60 + 1e-7
    assert float((sleeves @ target).mean()) >= -1e-7


def test_infeasible_paper_return_floor_fails_closed():
    sleeves, market = synthetic_fit()
    sleeves = sleeves - 0.02
    market = market + 0.02
    previous = pd.Series(0.2, index=sleeves.columns)
    with pytest.raises(SleeveAllocationError, match="NO_FEASIBLE_ALLOCATION_SOLUTION"):
        solve_sleeve_allocation(
            sleeves,
            market,
            previous,
            SleeveAllocatorConfig(return_floor="spy"),
        )


def test_first_target_only_earns_following_session_and_pays_funding_cost():
    sleeves, market = synthetic_fit(160)
    config = SleeveAllocatorConfig(
        objective_mode="equal", return_floor="none", fit_window=126, hold_window=63
    )
    result = run_sleeve_allocation(sleeves, market, config, common_warmup=126)
    assert result.status == "COMPLETED"
    decision = sleeves.index[125]
    first_live = sleeves.index[126]
    assert result.daily.loc[decision, "net_return"] == 0
    assert result.daily.loc[first_live, "allocation_cost"] == pytest.approx(0.0005)
    assert result.weights.loc[decision].abs().sum() == 0
    assert result.weights.loc[first_live].sum() == pytest.approx(1)
    expected = sleeves.loc[first_live].mean() - 0.0005
    assert result.daily.loc[first_live, "net_return"] == pytest.approx(expected)


def test_one_session_implementation_delay_moves_first_live_return():
    sleeves, market = synthetic_fit(160)
    config = SleeveAllocatorConfig(
        objective_mode="equal", return_floor="none", fit_window=126, hold_window=63
    )
    result = run_sleeve_allocation(
        sleeves, market, config, common_warmup=126, implementation_delay_sessions=1
    )
    assert result.status == "COMPLETED"
    assert result.weights.loc[sleeves.index[126]].abs().sum() == 0
    assert result.weights.loc[sleeves.index[127]].sum() == pytest.approx(1)
    expected = sleeves.loc[sleeves.index[127]].mean() - 0.0005
    assert result.daily.loc[sleeves.index[127], "net_return"] == pytest.approx(expected)

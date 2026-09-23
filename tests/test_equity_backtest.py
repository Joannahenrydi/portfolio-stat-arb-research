import numpy as np
import pandas as pd

from backtest.engine import (
    FastPortfolioConfig,
    candidate_scores,
    prior_market_beta,
    run_fast_backtest,
)


def test_beta_is_prior_only_and_scores_have_declared_names():
    dates = pd.bdate_range("2020-01-01", periods=300)
    market = pd.Series(np.linspace(-0.01, 0.01, len(dates)), index=dates)
    returns = pd.DataFrame({"a": market, "b": -market}, index=dates)
    beta = prior_market_beta(returns, market, 20)
    changed = returns.copy()
    changed.iloc[-1] = [0.9, -0.9]
    actual = prior_market_beta(changed, market, 20)
    pd.testing.assert_series_equal(beta.iloc[-1], actual.iloc[-1])
    assert set(candidate_scores(returns)) == {
        "reversal_1", "reversal_3", "reversal_5", "reversal_10",
        "momentum_21_skip_5", "momentum_63_skip_5", "momentum_126_skip_21",
        "momentum_252_skip_21", "blend_rev3_mom63",
    }


def test_signal_at_t_cannot_earn_same_day_return_and_costs_are_charged():
    dates = pd.bdate_range("2020-01-01", periods=150)
    names = pd.Index([f"s{i}" for i in range(60)])
    returns = pd.DataFrame(0.0, index=dates, columns=names)
    returns.iloc[-1, :30] = 0.1
    score = pd.DataFrame(np.tile(np.r_[np.ones(30), -np.ones(30)], (len(dates), 1)), index=dates, columns=names)
    beta = pd.DataFrame(1.0, index=dates, columns=names)
    sectors = pd.Series(np.tile(["a", "b"], 30), index=names)
    close = pd.DataFrame(100.0, index=dates, columns=names)
    volume = pd.DataFrame(1_000_000.0, index=dates, columns=names)
    result, weights = run_fast_backtest(
        score, returns, beta, sectors, close, volume,
        config=FastPortfolioConfig(rebalance_every=60),
    )
    assert result.iloc[0].gross_return == 0
    assert result.transaction_cost.sum() > 0
    expected = float((weights.iloc[-1] * returns.iloc[-1]).sum())
    assert result.iloc[-1].gross_return == expected


def test_holdings_drift_between_rebalances_instead_of_free_daily_rebalancing():
    dates = pd.bdate_range("2020-01-01", periods=65)
    names = pd.Index([f"s{i}" for i in range(60)])
    returns = pd.DataFrame(0.0, index=dates, columns=names)
    returns.iloc[-2, 0] = 0.10
    score = pd.DataFrame(
        np.tile(np.linspace(-1, 1, len(names)), (len(dates), 1)), index=dates, columns=names
    )
    beta = pd.DataFrame(1.0, index=dates, columns=names)
    sectors = pd.Series(np.tile(["a", "b"], 30), index=names)
    close = pd.DataFrame(100.0, index=dates, columns=names)
    volume = pd.DataFrame(1_000_000.0, index=dates, columns=names)
    _, weights = run_fast_backtest(
        score, returns, beta, sectors, close, volume,
        config=FastPortfolioConfig(rebalance_every=60),
    )
    assert weights.iloc[-1, 0] != weights.iloc[-2, 0]

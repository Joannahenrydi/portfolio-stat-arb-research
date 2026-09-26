import numpy as np
import pandas as pd

from backtest.cross_asset_budget import RiskBudgetConfig, run_risk_budget_backtest


def test_risk_budget_targets_respect_all_caps():
    rng = np.random.default_rng(41)
    index = pd.bdate_range("2018-01-02", periods=200)
    columns = [f"A{i}" for i in range(12)]
    returns = pd.DataFrame(rng.normal(0, .004, (200, 12)), index=index, columns=columns)
    alpha = pd.DataFrame(rng.normal(0, .002, (200, 12)), index=index, columns=columns)
    adv = pd.DataFrame(100_000_000.0, index=index, columns=columns)
    eligibility = pd.DataFrame(True, index=index, columns=columns)
    factors = pd.DataFrame({
        "equity": [1] * 4 + [0] * 8,
        "duration": [0] * 4 + [1] * 4 + [0] * 4,
        "commodity": [0] * 8 + [1] * 4,
    }, index=columns)
    sleeves = pd.Series(["equity"] * 4 + ["rates"] * 4 + ["commodity"] * 4,
                        index=columns)
    result = run_risk_budget_backtest(
        alpha, returns, adv, eligibility, factors,
        pd.Series({"equity": .12, "duration": .12, "commodity": .12}),
        sleeves, pd.Series({"equity": .25, "rates": .25, "commodity": .25}),
        config=RiskBudgetConfig(covariance_window=60, net_cap=.20),
    )
    assert result.status == "COMPLETED"
    assert result.rebalances.net_budget_ratio.max() <= 1.0001
    assert result.rebalances.maximum_factor_budget_ratio.max() <= 1.0001
    assert result.rebalances.maximum_sleeve_budget_ratio.max() <= 1.0001
    assert result.daily.transaction_cost.sum() > 0


def test_risk_scaler_reduces_realized_gross_budget():
    rng = np.random.default_rng(42)
    index = pd.bdate_range("2018-01-02", periods=160)
    columns = [f"A{i}" for i in range(12)]
    returns = pd.DataFrame(rng.normal(0, .004, (160, 12)), index=index, columns=columns)
    alpha = pd.DataFrame(rng.normal(0, .002, (160, 12)), index=index, columns=columns)
    adv = pd.DataFrame(100_000_000.0, index=index, columns=columns)
    factors = pd.DataFrame({"factor": [1, -1] * 6}, index=columns)
    sleeves = pd.Series(["left"] * 6 + ["right"] * 6, index=columns)
    scaler = pd.Series(.5, index=index)
    result = run_risk_budget_backtest(
        alpha, returns, adv, pd.DataFrame(True, index=index, columns=columns), factors,
        pd.Series({"factor": .5}), sleeves, pd.Series({"left": .5, "right": .5}),
        risk_scaler=scaler, config=RiskBudgetConfig(covariance_window=60, net_cap=.5),
    )
    assert result.status == "COMPLETED"
    assert result.rebalances.gross.max() <= .5001
    assert result.rebalances.risk_scaler.eq(.5).all()


def test_temporarily_ineligible_holding_is_carried_without_zero_adv_liquidation():
    rng = np.random.default_rng(43)
    index = pd.bdate_range("2018-01-02", periods=120)
    columns = [f"A{i}" for i in range(12)]
    returns = pd.DataFrame(rng.normal(0, .003, (120, 12)), index=index, columns=columns)
    alpha = pd.DataFrame(0.002, index=index, columns=columns)
    alpha.iloc[:, 1::2] *= -1
    adv = pd.DataFrame(100_000_000.0, index=index, columns=columns)
    eligibility = pd.DataFrame(True, index=index, columns=columns)
    # After positions exist, A0 cannot be traded at one rebalance but still has a valid return.
    eligibility.loc[index[100], "A0"] = False
    adv.loc[index[100], "A0"] = 0
    factors = pd.DataFrame({"factor": [1, -1] * 6}, index=columns)
    sleeves = pd.Series(["all"] * 12, index=columns)
    result = run_risk_budget_backtest(
        alpha, returns, adv, eligibility, factors, pd.Series({"factor": .5}),
        sleeves, pd.Series({"all": 1.0}),
        config=RiskBudgetConfig(covariance_window=60, net_cap=.5),
    )
    assert result.status == "COMPLETED"
    assert result.daily.transaction_cost.notna().all()

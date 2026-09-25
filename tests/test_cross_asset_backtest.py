import numpy as np
import pandas as pd

from backtest.cross_asset import CrossAssetConfig, run_cross_asset_backtest


def test_cross_asset_targets_are_factor_neutral_and_costed():
    rng = np.random.default_rng(31)
    index = pd.bdate_range("2018-01-02", periods=180)
    columns = [f"A{i}" for i in range(10)]
    returns = pd.DataFrame(rng.normal(0, 0.005, (180, 10)), index=index, columns=columns)
    alpha = pd.DataFrame(rng.normal(0, 0.002, (180, 10)), index=index, columns=columns)
    adv = pd.DataFrame(100_000_000.0, index=index, columns=columns)
    eligibility = pd.DataFrame(True, index=index, columns=columns)
    factors = pd.DataFrame(
        {"equity": [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
         "duration": [0, 0, 0, 1, 2, 3, 0, 0, 0, 0],
         "commodity": [0, 0, 0, 0, 0, 0, 1, 1, -1, -1]},
        index=columns,
    )
    result = run_cross_asset_backtest(
        alpha, returns, adv, eligibility, factors,
        config=CrossAssetConfig(covariance_window=60),
    )
    assert result.status == "COMPLETED"
    assert result.rebalances.net.abs().max() < 1e-8
    assert result.rebalances.max_factor_exposure.abs().max() < 1e-8
    assert result.daily.transaction_cost.sum() > 0

import unittest

import numpy as np
import pandas as pd

from pairs_trading.cli import synthetic_prices
from pairs_trading.dynamic import (
    DynamicParameters,
    dynamic_signals,
    optimal_x_weight,
    simulate_dynamic,
)
from pairs_trading.research import Costs


class DynamicTests(unittest.TestCase):
    def test_drawdown_breaker_exits_next_bar_and_stays_flat(self):
        dates = pd.bdate_range("2020-01-01", periods=8)
        market = pd.DataFrame({"X": 10., "Y": [20., 20., 9., 9., 9., 9., 9., 9.],
                               "div_X": 0., "div_Y": 0.}, index=dates)
        signals = pd.DataFrame({"prediction": .1, "weight_x": .5, "weight_y": .5,
                                "sigma_daily": .001, "correlation": .8, "volatility_ratio": 1.,
                                "var_x": .0001, "var_y": .0001, "covariance": .00008}, index=dates)
        costs = Costs(0, 0, 0, 0, 0)
        summary, curve, trades = simulate_dynamic(market, signals, DynamicParameters(horizon=10),
                                                  "2020-01-01", "2020-02-01", capital=1000, costs=costs)
        self.assertTrue(summary["stopped"])
        self.assertEqual(summary["entries"], 1)
        self.assertEqual(trades.iloc[1].date, dates[3])
        self.assertTrue((curve.gross.iloc[3:] == 0).all())

    def test_hedge_solution_minimizes_stated_objective(self):
        vx, vy, cov, previous = .0004, .0001, .00005, .53
        actual = optimal_x_weight(vx, vy, cov, previous)
        a = np.linspace(.4, .6, 10001)
        trace = vx+vy
        objective = (a*a*vx+(1-a)**2*vy-2*a*(1-a)*cov
                     + .1*trace*(a-.5)**2+.5*trace*(a-previous)**2)
        self.assertAlmostEqual(actual, a[np.argmin(objective)], places=4)

    def test_training_labels_mature_before_fit_and_future_invariance(self):
        market = synthetic_prices(850).rename(columns={"PAIR_X": "X", "PAIR_Y": "Y"})
        market["div_X"] = market["div_Y"] = 0.
        params = DynamicParameters(train_window=252, horizon=10)
        signals, fits = dynamic_signals(market, params)
        self.assertGreater(len(fits), 10)
        self.assertTrue((fits.last_label_maturity <= fits.date).all())
        changed = market.copy()
        changed.loc[changed.index[650]:, "X"] *= 1.3
        changed.loc[changed.index[650]:, "div_Y"] = 1.
        modified, _ = dynamic_signals(changed, params)
        np.testing.assert_allclose(signals.iloc[:650], modified.iloc[:650], equal_nan=True)
        prefix, _ = dynamic_signals(market.iloc[:650], params)
        np.testing.assert_allclose(signals.iloc[:650], prefix, equal_nan=True)
        kalman_params = DynamicParameters(train_window=252, horizon=10, features="kalman")
        original, _ = dynamic_signals(market, kalman_params)
        shorter, _ = dynamic_signals(market.iloc[:650], kalman_params)
        np.testing.assert_allclose(original.iloc[180:650], shorter.iloc[180:], equal_nan=True)

    def test_prior_close_shares_dividends_costs_and_exit(self):
        dates = pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-06", "2020-01-07"])
        market = pd.DataFrame({"X": [10., 10., 10., 11.], "Y": 20.,
                               "div_X": [0., 0., 1., 0.], "div_Y": [0., 0., 1., 0.]}, index=dates)
        signals = pd.DataFrame({"prediction": [.1, .1, -.1, -.1], "weight_x": .5,
                                "weight_y": .5, "sigma_daily": .001, "correlation": .8,
                                "volatility_ratio": 1., "var_x": .0001,
                                "var_y": .0001, "covariance": .00008}, index=dates)
        signals.loc[dates[1], "weight_x"] = .6  # Must not change today's already scheduled shares.
        costs = Costs(commission_per_share=0, minimum_per_order=1, slippage_bps=0,
                      sell_fee_bps=0, borrow_annual=.365)
        summary, curve, trades = simulate_dynamic(market, signals, DynamicParameters(confidence=0),
                                                  "2020-01-02", "2020-01-07", capital=1000, costs=costs)
        self.assertEqual(trades.iloc[0].signal_date, dates[0])
        self.assertEqual(trades.iloc[0].date, dates[1])
        self.assertEqual(trades.iloc[0].qx, -10)
        self.assertEqual(trades.iloc[0].qy, 5)
        self.assertAlmostEqual(summary["final_equity"], 980.6)
        self.assertAlmostEqual(summary["borrow"], .4)
        self.assertEqual(curve.iloc[-1].gross, 0)


if __name__ == "__main__":
    unittest.main()

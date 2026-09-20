import unittest

import numpy as np
import pandas as pd

from pairs_trading.cli import synthetic_prices
from pairs_trading.research import Costs, Parameters, research_signals, simulate


class ResearchTests(unittest.TestCase):
    def test_actual_share_hedge_dividends_calendar_borrow_and_commissions(self):
        dates = pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-06", "2020-01-07"])
        market = pd.DataFrame({"X": [10., 10., 10., 11.], "Y": 20.,
                               "div_X": [0., 0., 1., 0.], "div_Y": [0., 0., 1., 0.]}, index=dates)
        signals = pd.DataFrame({"z": [-2., -2., 0., 0.], "hedge": 2., "edge": 100.}, index=dates)
        signals.loc[dates[1], "hedge"] = 99.  # Fill must use yesterday's ratio of 2.
        costs = Costs(commission_per_share=0, minimum_per_order=1, slippage_bps=0,
                      sell_fee_bps=0, borrow_annual=.365)
        summary, curve, trades = simulate(market, signals, Parameters(), "2020-01-02", "2020-01-07",
                                          capital=1000, costs=costs)
        self.assertEqual(trades.iloc[0].date, dates[1])
        self.assertEqual(trades.iloc[0].qx, -10)
        self.assertEqual(trades.iloc[0].qy, 5)
        self.assertAlmostEqual(summary["commission"], 4)
        self.assertAlmostEqual(summary["borrow"], .4)
        self.assertAlmostEqual(summary["dividends_net"], -5)
        self.assertAlmostEqual(summary["final_equity"], 980.6)
        self.assertEqual(curve.iloc[-1].gross, 0)

    def test_future_prices_and_dividends_cannot_change_past_signals(self):
        market = synthetic_prices(500).rename(columns={"PAIR_X": "X", "PAIR_Y": "Y"})
        market["div_X"] = market["div_Y"] = 0.
        changed = market.copy()
        changed.loc[changed.index[400]:, "Y"] *= 2
        changed.loc[changed.index[400]:, "div_X"] = 1
        for model in ("kalman", "rolling_ols"):
            p = Parameters(model=model)
            original = research_signals(market, p)
            modified = research_signals(changed, p)
            np.testing.assert_allclose(original.iloc[200:400], modified.iloc[200:400], equal_nan=True)

    def test_cash_baseline_never_trades(self):
        market = synthetic_prices(400).rename(columns={"PAIR_X": "X", "PAIR_Y": "Y"})
        market["div_X"] = market["div_Y"] = 0.
        p = Parameters()
        summary, _curve, trades = simulate(market, research_signals(market, p), p,
                                          "2021-01-01", "2022-12-31", fraction=0)
        self.assertEqual(summary["return"], 0)
        self.assertEqual(summary["entries"], 0)
        self.assertTrue(trades.empty)

    def test_expensive_small_orders_fail_cost_filter(self):
        dates = pd.bdate_range("2020-01-01", periods=8)
        market = pd.DataFrame({"X": 10., "Y": 20., "div_X": 0., "div_Y": 0.}, index=dates)
        signals = pd.DataFrame({"z": -2., "hedge": 2., "edge": 1.}, index=dates)
        result = simulate(market, signals, Parameters(), "2020-01-01", "2020-02-01",
                          capital=1000, costs=Costs(minimum_per_order=100))
        self.assertEqual(result[0]["entries"], 0)


if __name__ == "__main__":
    unittest.main()

import unittest

import numpy as np

from pairs_trading.backtest import run_backtest
from pairs_trading.broker import AlpacaPaperBroker
from pairs_trading.cli import synthetic_prices
from pairs_trading.model import KalmanHedgeRatio, StrategyConfig, build_signals


class StrategyTests(unittest.TestCase):
    def test_kalman_converges_to_ratio(self):
        filt = KalmanHedgeRatio(0.5, 1e-6, 0.01)
        for x in np.linspace(10, 100, 100):
            filt.update(float(x), float(1.7 * x))
        self.assertAlmostEqual(filt.beta, 1.7, places=2)

    def test_signal_has_no_same_bar_lookahead(self):
        cfg = StrategyConfig(formation_window=30, z_window=20)
        prices = synthetic_prices(120)
        original = build_signals(prices, "PAIR_X", "PAIR_Y", cfg)
        self.assertTrue(original.zscore.iloc[:30].isna().all())
        changed = prices.copy()
        changed.iloc[-1, 1] *= 2
        recalculated = build_signals(changed, "PAIR_X", "PAIR_Y", cfg)
        self.assertEqual(original.zscore.iloc[-2], recalculated.zscore.iloc[-2])

    def test_backtest_outputs_finite_equity(self):
        result = run_backtest(synthetic_prices(), "PAIR_X", "PAIR_Y", StrategyConfig())
        self.assertTrue(np.isfinite(result.equity.equity).all())
        self.assertGreater(result.summary["entries"], 0)
        self.assertGreater(result.summary["final_equity"], 0)

    def test_invalid_prices_rejected(self):
        prices = synthetic_prices(220)
        prices.iloc[0, 0] = 0
        with self.assertRaises(ValueError):
            build_signals(prices, "PAIR_X", "PAIR_Y", StrategyConfig(30, 20))

    def test_holdout_warms_filter_without_early_trades(self):
        prices = synthetic_prices(400)
        start = str(prices.index[250].date())
        cfg = StrategyConfig()
        result = run_backtest(prices, "PAIR_X", "PAIR_Y", cfg, trade_start=start)
        self.assertEqual(result.equity.index[0], prices.index[250])
        self.assertEqual(result.equity.equity.iloc[0], cfg.initial_capital)
        self.assertGreater(len(result.trades), 0)
        self.assertTrue((result.trades.date > prices.index[250]).all())
        expected = build_signals(prices, "PAIR_X", "PAIR_Y", cfg)
        np.testing.assert_allclose(result.signals.beta, expected.beta)

    def test_insufficient_capital_does_not_create_orphan_leg(self):
        prices = synthetic_prices(400)
        prices["PAIR_Y"] *= 1000
        cfg = StrategyConfig(initial_capital=1000)
        result = run_backtest(prices, "PAIR_X", "PAIR_Y", cfg)
        self.assertTrue((result.equity.qx == 0).all())
        self.assertTrue((result.equity.qy == 0).all())
        self.assertTrue((result.equity.equity == 1000).all())

    def test_broker_recognizes_pair_and_rejects_orphan(self):
        class Position:
            def __init__(self, symbol, qty):
                self.symbol, self.qty = symbol, qty

        class Trading:
            def __init__(self):
                self.positions = [Position("X", "-10"), Position("Y", "7")]

            def get_all_positions(self):
                return self.positions

        broker = object.__new__(AlpacaPaperBroker)
        broker.trading = Trading()
        self.assertEqual(broker.pair_position("X", "Y"), 1)
        broker.trading.positions = [Position("X", "-10")]
        with self.assertRaises(RuntimeError):
            broker.pair_position("X", "Y")


if __name__ == "__main__":
    unittest.main()

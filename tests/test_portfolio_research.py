import numpy as np
import pandas as pd
import pytest

from pairs_trading.portfolio_research import build_features, neutral_target, simulate, trading_cost


def fixture():
    rng = np.random.default_rng(41)
    dates = pd.bdate_range("2017-01-01", periods=700)
    symbols = list("ABCDEFGH")
    config = {
        "groups": dict(zip(symbols, ["a", "a", "b", "b", "c", "c", "d", "d"])),
        "market": "SPY",
    }
    prices = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, (700, 9)), axis=0))
    frames = []
    for j, s in enumerate(symbols + ["SPY"]):
        frames.append(
            pd.DataFrame(
                {
                    "date": dates,
                    "symbol": s,
                    "adjusted_close": prices[:, j],
                    "raw_close_proxy": prices[:, j],
                    "volume": 1e7,
                    "split": 0,
                    "dividend": 0,
                }
            )
        )
    return pd.concat(frames), config


def test_no_future_prices_affect_features():
    panel, c = fixture()
    f = build_features(panel, c)
    cutoff = sorted(panel.date.unique())[500]
    shorter = build_features(panel[panel.date <= cutoff], c)
    for key in ["beta", "variance", "adv", "eligible"]:
        pd.testing.assert_frame_equal(f[key].loc[:cutoff], shorter[key])
    for key in f["signals"]:
        pd.testing.assert_frame_equal(f["signals"][key].loc[:cutoff], shorter["signals"][key])


def test_neutrality_and_caps():
    score = np.array([1, -2, 3, 1, 2, -1, 3, -4.0])
    beta = np.array([0.7, 0.9, 1, 1.2, 0.8, 0.9, 1.1, 1.3])
    g = ["a", "a", "b", "b", "c", "c", "d", "d"]
    w = neutral_target(score, beta, np.full(8, 0.001), g)
    assert np.abs(w).sum() <= 1 + 1e-10
    assert np.abs(w).max() <= 0.1 + 1e-10
    assert abs(w @ beta) < 1e-10
    for sector in set(g):
        assert abs(w[np.array(g) == sector].sum()) < 1e-10


def test_cost_and_missing_cost_guard():
    args = (np.array([0.1, -0.1]), np.array([0.02, 0.02]), np.array([1e7, 1e7]), 1e5)
    assert trading_cost(*args).sum() > 0
    np.testing.assert_allclose(trading_cost(*args, multiplier=2), 2 * trading_cost(*args))
    with pytest.raises(ValueError, match="missing cost"):
        trading_cost(args[0], args[1], np.array([np.nan, 1e7]), 1e5)


def test_first_day_is_cost_only_and_terminal_cash():
    panel, c = fixture()
    f = build_features(panel, c)
    dates = f["returns"].index
    daily, w, attrib = simulate(f, c, start=str(dates[400].date()), end=str(dates[440].date()))
    assert daily.iloc[0].gross_return == 0
    assert daily.iloc[0].net_return <= 0
    assert np.abs(w.iloc[-1]).sum() == 0
    np.testing.assert_allclose(attrib.sum(axis=1), daily.net_return, atol=1e-12)
    assert daily.net_exposure.abs().max() < 1e-9
    assert daily.beta_exposure.abs().max() < 1e-9


def test_missing_held_return_is_not_silently_zeroed():
    panel, c = fixture()
    f = build_features(panel, c)
    dates = f["returns"].index
    f["returns"].loc[dates[402], :] = np.nan
    with pytest.raises(ValueError, match="missing held return"):
        simulate(f, c, start=str(dates[400].date()), end=str(dates[440].date()))

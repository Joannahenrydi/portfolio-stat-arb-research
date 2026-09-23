import time

import numpy as np
import pandas as pd

from backtest.attribution import attribute_pnl
from backtest.walk_forward import expanding_walk_forward
from features.alphas import forward_total_return_labels
from live.paper import build_order_plan, submit_paper_orders
from portfolio.neutralization import build_exposure_matrix, exposure_report
from portfolio.optimizer import PortfolioConstraints, construct_portfolio
from risk.promotion import evaluate_promotion


def test_optimizer_handles_400_name_neutral_portfolio():
    rng = np.random.default_rng(4)
    n = 400
    names = pd.Index([f"sid-{i:03d}" for i in range(n)])
    alpha = pd.Series(rng.normal(0, 0.003, n), index=names)
    factor = rng.normal(size=(n, 8))
    covariance = pd.DataFrame(
        factor @ factor.T * 1e-6 + np.eye(n) * 0.0003, index=names, columns=names
    )
    beta = pd.Series(rng.uniform(0.7, 1.3, n), index=names)
    sectors = pd.Series(np.repeat([f"s{i}" for i in range(10)], 40), index=names)
    exposure = build_exposure_matrix(beta, sectors)
    start = time.perf_counter()
    result = construct_portfolio(
        alpha,
        covariance,
        exposure,
        pd.Series(0.0, index=names),
        pd.Series(50_000_000.0, index=names),
        1_000_000.0,
        constraints=PortfolioConstraints(max_name=0.01, max_turnover=1.0),
    )
    elapsed = time.perf_counter() - start
    assert result.status == "ACCEPTED", result.metrics
    assert elapsed < 10
    assert exposure_report(result.target, exposure).abs().max() < 1e-7
    assert result.metrics["gross"] <= 1.0


def test_walk_forward_predictions_do_not_change_when_future_labels_change():
    rng = np.random.default_rng(2)
    dates = pd.bdate_range("2020-01-01", periods=80)
    names = [f"sid-{i}" for i in range(8)]
    signal = pd.DataFrame(rng.normal(size=(80, 8)), index=dates, columns=names)
    returns = signal.shift(1) * 0.001 + pd.DataFrame(
        rng.normal(scale=0.005, size=(80, 8)), index=dates, columns=names
    )
    labels, label_end = forward_total_return_labels(returns, 1)
    kwargs = {
        "first_prediction": dates[40],
        "refit_every": 10,
        "pit_verified": False,
        "input_scope": "research_snapshot_only",
        "min_samples": 100,
    }
    original = expanding_walk_forward({"signal": signal}, labels, label_end, **kwargs)
    changed = labels.copy()
    changed.loc[dates[60]:] += 1.0
    replay = expanding_walk_forward({"signal": signal}, changed, label_end, **kwargs)
    pd.testing.assert_frame_equal(
        original.predictions.loc[: dates[59]], replay.predictions.loc[: dates[59]]
    )
    assert all(a.latest_label_session <= a.train_end for a in original.artifacts)


def test_promotion_gate_order_plan_and_attribution_are_fail_closed():
    rejected = evaluate_promotion(
        {"sharpe": 0.8, "max_drawdown": -0.1},
        {"sharpe": 0.2, "max_drawdown": -0.1},
        paper_sessions=60,
    )
    targets = pd.Series({"A": 0.1, "B": -0.1})
    shares = pd.Series({"A": 0, "B": 0})
    prices = pd.Series({"A": 100.0, "B": 50.0})
    assert rejected.status == "REJECTED"
    assert build_order_plan(targets, shares, prices, 100_000, orders_allowed=False) == []
    assert submit_paper_orders([], orders_allowed=False, submission_enabled=True) == []
    accepted = evaluate_promotion(
        {"sharpe": 0.8, "max_drawdown": -0.1},
        {"sharpe": 0.7, "max_drawdown": -0.1},
        paper_sessions=60,
    )
    orders = build_order_plan(targets, shares, prices, 100_000, orders_allowed=accepted.orders_allowed)
    assert [(order.symbol, order.quantity) for order in orders] == [("A", 100), ("B", -200)]
    contributions, ledger = attribute_pnl(
        targets, pd.Series({"A": 0.01, "B": -0.02}), transaction_cost=0.0001
    )
    assert ledger["gross_return"] == contributions.sum()
    assert ledger["net_return"] == ledger["gross_return"] - 0.0001


def test_nonfinite_metrics_cannot_pass_promotion():
    decision = evaluate_promotion(
        {"sharpe": np.nan, "max_drawdown": -0.1},
        {"sharpe": 0.8, "max_drawdown": np.nan},
        paper_sessions=60,
    )
    assert decision.status == "REJECTED"
    assert decision.orders_allowed is False

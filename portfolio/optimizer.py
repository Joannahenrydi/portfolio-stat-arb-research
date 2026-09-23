"""Cost-aware equity portfolio targets; this module never assumes or submits fills.

All input and returned weights are notionals divided by *current, pretrade* NAV.
Expected alpha is a decimal return over the same horizon as ``calendar_days``.
The optimizer reserves transaction costs and the projected calendar-day borrow
charge, so actual weights after that reserve also satisfy the portfolio caps.
BLOCKED/REJECTED results preserve existing holdings: cash is never an assumed
escape from an infeasible liquidation. Data/PIT and promotion gates live above
this numerical module; ACCEPTED means only that this target is feasible.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import NamedTuple

import numpy as np
import pandas as pd
from scipy.linalg import qr
from scipy.optimize import Bounds, LinearConstraint, linprog, minimize


@dataclass(frozen=True)
class PortfolioCosts:
    commission_bps: float = 0.5
    half_spread_bps: float = 2.0
    slippage_bps: float = 1.0
    impact_coefficient: float = 0.10
    borrow_annual: float = 0.03
    calendar_days: float = 1.0
    multiplier: float = 1.0


@dataclass(frozen=True)
class PortfolioConstraints:
    max_gross: float = 1.0
    max_name: float = 0.01
    max_turnover: float = 0.25  # sum(abs(trade notional)) / current NAV
    max_participation: float = 0.001
    risk_aversion: float = 5.0
    tolerance: float = 1e-7
    max_iterations: int = 500


class PortfolioResult(NamedTuple):
    target: pd.Series
    metrics: dict
    status: str


def _validate_settings(costs, constraints=None):
    for field in fields(costs):
        value = getattr(costs, field.name)
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"{field.name} must be finite and nonnegative")
    if constraints is not None:
        for field in fields(constraints):
            value = getattr(constraints, field.name)
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{field.name} must be finite and nonnegative")
        if constraints.tolerance <= 0 or constraints.max_iterations < 1:
            raise ValueError("positive tolerance and iteration budget required")


def estimate_costs(previous_weights, target_weights, daily_volatility, adv, nav, costs=None):
    """Return projected costs as fractions of pretrade NAV, including calendar borrow.

    Inputs are equally ordered one-dimensional arrays. Impact per traded dollar
    is coefficient * daily volatility * sqrt(order dollars / dollar ADV).
    Borrow is a *projected* charge on target shorts, not a historical cash ledger.
    Zero ADV permits holding a position but never permits a modeled trade.
    """
    costs = costs or PortfolioCosts()
    _validate_settings(costs)
    previous, target, vol, dollar_adv = [
        np.asarray(a, dtype=float)
        for a in (previous_weights, target_weights, daily_volatility, adv)
    ]
    if previous.ndim != 1 or not all(a.shape == previous.shape for a in (target, vol, dollar_adv)):
        raise ValueError("cost inputs must be equally sized one-dimensional arrays")
    if not all(np.isfinite(a).all() for a in (previous, target, vol, dollar_adv)):
        raise ValueError("cost inputs must be finite")
    if not np.isfinite(nav) or nav <= 0 or (vol < 0).any() or (dollar_adv < 0).any():
        raise ValueError("positive NAV and nonnegative volatility/ADV required")
    delta = np.abs(target - previous)
    if ((delta > 0) & (dollar_adv == 0)).any():
        raise ValueError("cannot price a trade with zero ADV")
    participation = np.divide(delta * nav, dollar_adv, out=np.zeros_like(delta), where=dollar_adv > 0)
    commission = delta.sum() * costs.commission_bps / 10000
    spread = delta.sum() * costs.half_spread_bps / 10000
    slippage = delta.sum() * costs.slippage_bps / 10000
    impact = float((delta * costs.impact_coefficient * vol * np.sqrt(participation)).sum())
    borrow = float(np.maximum(-target, 0).sum()) * costs.borrow_annual * costs.calendar_days / 365
    components = {
        "commission": commission,
        "spread": spread,
        "slippage": slippage,
        "impact": impact,
        "borrow": borrow,
    }
    components = {name: float(value * costs.multiplier) for name, value in components.items()}
    components["transaction"] = sum(components[k] for k in ("commission", "spread", "slippage", "impact"))
    components["total"] = components["transaction"] + components["borrow"]
    return components


def _aligned_series(value, index, name):
    if not isinstance(value, pd.Series) or not value.index.is_unique:
        raise ValueError(f"{name} must be a Series with unique security IDs")
    if len(value) != len(index) or set(value.index) != set(index):
        raise ValueError(f"{name} security IDs do not match alpha")
    return value.reindex(index).to_numpy(dtype=float)


def _construct_portfolio_slsqp_reference(
    alpha, covariance, exposure, prev_weights, adv, nav, costs=None, constraints=None
):
    """Solve a convex alpha-minus-risk-minus-conservative-cost target problem.

    ``alpha``, ``prev_weights`` and ``adv`` are Series indexed by stable security
    ID; ADV is dollars/day. Covariance is a labeled PSD DataFrame in daily return
    units (or horizon units matching alpha). Impact volatility uses its diagonal,
    so pass daily covariance when alpha is daily. Every column of ``exposure``
    is constrained to zero (beta, sector dummies, optional factors). Net dollar
    neutrality is always added. Redundant exposure columns are safely removed.

    The L1 turnover coefficient includes an upper bound on square-root impact
    over each name's allowed order size. Gross/name caps reserve this cost bound
    plus borrow before solving; exact projected costs are checked afterwards.
    ``target, metrics, status`` can be unpacked from the result. Only ACCEPTED
    targets are plans; higher-level data/OOS/paper approval remains necessary.
    """
    costs = costs or PortfolioCosts()
    limits = constraints or PortfolioConstraints()
    _validate_settings(costs, limits)
    if not isinstance(alpha, pd.Series) or not alpha.index.is_unique or alpha.empty:
        raise ValueError("alpha must be a nonempty Series with unique security IDs")
    index = alpha.index
    n = len(index)
    previous = _aligned_series(prev_weights, index, "prev_weights")
    dollar_adv = _aligned_series(adv, index, "adv")
    forecasts = alpha.to_numpy(dtype=float)
    if not np.isfinite(previous).all() or not np.isfinite(nav) or nav <= 0:
        raise ValueError("finite current weights and positive NAV required")
    if not isinstance(covariance, pd.DataFrame):
        raise TypeError("covariance must be a labeled DataFrame")
    if (not covariance.index.is_unique or not covariance.columns.is_unique
            or set(covariance.index) != set(index) or set(covariance.columns) != set(index)):
        raise ValueError("covariance security IDs do not match alpha")
    cov = covariance.reindex(index=index, columns=index).to_numpy(dtype=float)
    if not np.isfinite(cov).all() or not np.allclose(cov, cov.T, atol=1e-12, rtol=1e-8):
        raise ValueError("covariance must be finite and symmetric")
    eigenvalues = np.linalg.eigvalsh(cov)
    if eigenvalues.min() < -1e-12 * max(1.0, float(np.abs(eigenvalues).max())):
        raise ValueError("covariance must be positive semidefinite")
    # Remove only floating-point asymmetry and negligible negative eigenvalues.
    cov = (cov + cov.T) / 2
    cov += np.eye(n) * max(0.0, -float(eigenvalues.min()))
    if exposure is None:
        supplied = np.empty((n, 0))
    else:
        if (not isinstance(exposure, pd.DataFrame) or not exposure.index.is_unique
                or not exposure.columns.is_unique or set(exposure.index) != set(index)):
            raise ValueError("exposure must be a DataFrame with matching unique security IDs")
        supplied = exposure.reindex(index).to_numpy(dtype=float)
        if not np.isfinite(supplied).all():
            raise ValueError("exposure must be finite")
    all_exposure = np.column_stack([np.ones(n), supplied])

    def unchanged(status, reason, **extra):
        current_exposure = all_exposure.T @ previous
        violations = []
        if np.abs(previous).sum() > limits.max_gross + limits.tolerance:
            violations.append("gross")
        if np.abs(previous).max() > limits.max_name + limits.tolerance:
            violations.append("single_name")
        if np.abs(current_exposure).max() > limits.tolerance:
            violations.append("neutrality")
        return PortfolioResult(
            pd.Series(previous.copy(), index=index, name="target_weight"),
            {"reason": reason, "execution_allowed": False, "assumed_fills": 0,
             "turnover": 0.0, "current_constraint_violations": violations, **extra},
            status,
        )

    if not np.isfinite(forecasts).all():
        return unchanged("BLOCKED", "NONFINITE_ALPHA")
    if not np.isfinite(dollar_adv).all() or (dollar_adv < 0).any():
        return unchanged("BLOCKED", "MISSING_OR_INVALID_LIQUIDITY")
    if not np.any(forecasts):
        return unchanged("REJECTED", "NO_ALPHA")

    _, rmat, pivots = qr(all_exposure, mode="economic", pivoting=True)
    rank = np.linalg.matrix_rank(rmat)
    independent = all_exposure[:, pivots[:rank]].T
    # x = [weights, absolute trades, absolute holdings]. All constraints are linear.
    eye = np.eye(n)
    zeros = np.zeros((n, n))
    max_trade = np.minimum(limits.max_turnover, dollar_adv * limits.max_participation / nav)
    max_ratio = np.divide(max_trade * nav, dollar_adv, out=np.zeros(n), where=dollar_adv > 0)
    vol = np.sqrt(np.maximum(np.diag(cov), 0))
    trade_coefficient = costs.multiplier * (
        (costs.commission_bps + costs.half_spread_bps + costs.slippage_bps) / 10000
        + costs.impact_coefficient * vol * np.sqrt(max_ratio)
    )
    borrow_coefficient = costs.multiplier * costs.borrow_annual * costs.calendar_days / 365
    cost_vector = np.r_[np.full(n, -borrow_coefficient / 2), trade_coefficient,
                        np.full(n, borrow_coefficient / 2)]
    gross_row = np.r_[np.zeros(2 * n), np.ones(n)]
    turnover_row = np.r_[np.zeros(n), np.ones(n), np.zeros(n)]
    inequalities = np.vstack([
        np.hstack([eye, -eye, zeros]), np.hstack([-eye, -eye, zeros]),
        np.hstack([eye, zeros, -eye]), np.hstack([-eye, zeros, -eye]),
        gross_row + limits.max_gross * cost_vector,
        np.hstack([zeros, zeros, eye]) + limits.max_name * cost_vector,
        turnover_row, cost_vector,
    ])
    rhs = np.r_[previous, -previous, np.zeros(2 * n), limits.max_gross,
                np.full(n, limits.max_name), limits.max_turnover, 0.99]
    equalities = np.hstack([independent, np.zeros((rank, 2 * n))])
    lower = np.r_[np.full(n, -limits.max_name), np.zeros(2 * n)]
    upper = np.r_[np.full(n, limits.max_name), max_trade, np.full(n, limits.max_name)]
    bounds = list(zip(lower, upper))
    feasibility = linprog(
        cost_vector, A_ub=inequalities, b_ub=rhs, A_eq=equalities, b_eq=np.zeros(rank),
        bounds=bounds, method="highs",
        options={"primal_feasibility_tolerance": 1e-9, "dual_feasibility_tolerance": 1e-9},
    )
    if not feasibility.success:
        return unchanged("BLOCKED", "INFEASIBLE_CONSTRAINTS", solver_message=feasibility.message)

    # Eliminate the absolute-value auxiliaries for the quadratic solve. Keeping
    # them in a dense SLSQP problem creates 1,200 variables at 400 names and can
    # fail even on a simple diagonal covariance. The LP above still establishes
    # feasibility exactly; the smaller problem has the same convex objective.
    def reserve_cost(w):
        return float(trade_coefficient @ np.abs(w - previous)
                     + borrow_coefficient * np.maximum(-w, 0).sum())

    def reserve_gradient(w):
        return (trade_coefficient * np.sign(w - previous)
                + borrow_coefficient * (np.sign(w) - 1) / 2)

    def objective(w):
        return float((-forecasts @ w + limits.risk_aversion * (w @ cov @ w)
                      + reserve_cost(w)) * 100)

    def gradient(w):
        return (-forecasts + 2 * limits.risk_aversion * cov @ w
                + reserve_gradient(w)) * 100

    def budgets(w):
        remaining = 1 - reserve_cost(w)
        return np.r_[limits.max_gross * remaining - np.abs(w).sum(),
                     limits.max_name * remaining - np.abs(w),
                     limits.max_turnover - np.abs(w - previous).sum(),
                     remaining - 0.01]

    def budgets_gradient(w):
        cost_grad = reserve_gradient(w)
        return np.vstack([-limits.max_gross * cost_grad - np.sign(w),
                          -limits.max_name * cost_grad - np.diag(np.sign(w)),
                          -np.sign(w - previous), -cost_grad])

    weight_lower = np.maximum(-limits.max_name, previous - max_trade)
    weight_upper = np.minimum(limits.max_name, previous + max_trade)
    solution = minimize(
        objective, feasibility.x[:n], jac=gradient, method="SLSQP",
        bounds=Bounds(weight_lower, weight_upper),
        constraints=[{"type": "ineq", "fun": budgets, "jac": budgets_gradient},
                     LinearConstraint(independent, 0, 0)],
        options={"maxiter": int(limits.max_iterations), "ftol": 1e-11},
    )
    if not solution.success or not np.isfinite(solution.x).all():
        return unchanged("BLOCKED", "OPTIMIZER_FAILED", solver_message=solution.message)
    target = solution.x.copy()
    # Fixed zero-liquidity bounds must remain exactly unchanged despite roundoff.
    target[max_trade == 0] = previous[max_trade == 0]
    delta = np.abs(target - previous)
    projected = estimate_costs(previous, target, vol, dollar_adv, nav, costs)
    fraction = 1 - projected["total"]
    exposure_residual = float(np.abs(all_exposure.T @ target).max())
    reserve = reserve_cost(target)
    violations = []
    if fraction <= 0 or projected["total"] > reserve + limits.tolerance:
        violations.append("cost_reserve")
    else:
        if np.abs(target).sum() / fraction > limits.max_gross + limits.tolerance:
            violations.append("post_cost_gross")
        if np.abs(target).max() / fraction > limits.max_name + limits.tolerance:
            violations.append("post_cost_single_name")
        if exposure_residual / fraction > limits.tolerance:
            violations.append("post_cost_neutrality")
    if delta.sum() > limits.max_turnover + limits.tolerance:
        violations.append("turnover")
    if (delta > max_trade + limits.tolerance).any():
        violations.append("participation")
    if violations:
        return unchanged("BLOCKED", "POST_COST_CONSTRAINT_FAILURE", planned_violations=violations)
    if np.abs(previous).sum() <= limits.tolerance and np.abs(target).sum() <= limits.tolerance:
        return unchanged("REJECTED", "NO_ECONOMIC_TRADE")
    expected_return = float(forecasts @ target)
    variance = float(target @ cov @ target)
    metrics = {
        "reason": "FEASIBLE_TARGET_ONLY", "execution_allowed": True, "assumed_fills": 0,
        "turnover": float(delta.sum()), "gross": float(np.abs(target).sum()),
        "net": float(target.sum()), "max_name": float(np.abs(target).max()),
        "max_exposure_residual": exposure_residual,
        "post_cost_gross": float(np.abs(target).sum() / fraction),
        "post_cost_max_name": float(np.abs(target).max() / fraction),
        "expected_return": expected_return, "variance": variance,
        "expected_net_return": expected_return - projected["total"],
        "objective": expected_return - limits.risk_aversion * variance - reserve,
        "conservative_cost_reserve": reserve, "projected_costs": projected,
        "calendar_days": costs.calendar_days,
        "max_participation": float(np.divide(delta * nav, dollar_adv, out=np.zeros(n),
                                               where=dollar_adv > 0).max()),
        "solver_message": solution.message,
    }
    return PortfolioResult(pd.Series(target, index=index, name="target_weight"), metrics, "ACCEPTED")


def construct_portfolio(
    alpha, covariance, exposure, prev_weights, adv, nav, costs=None, constraints=None
):
    """Build a scalable covariance-aware market-neutral portfolio.

    The unconstrained mean-variance direction solves ``Sigma^-1 alpha`` and is
    projected into the null space of dollar, beta, sector, and supplied factor
    exposures. A one-dimensional exact-cost search then chooses how far to move
    from current holdings while enforcing gross, single-name, turnover, and ADV
    participation limits. This is a portfolio-level optimizer designed for
    hundreds of names; it does not perform trade-by-trade thresholding.
    """
    costs = costs or PortfolioCosts()
    limits = constraints or PortfolioConstraints()
    _validate_settings(costs, limits)
    if not isinstance(alpha, pd.Series) or alpha.empty or not alpha.index.is_unique:
        raise ValueError("alpha must be a nonempty Series with unique security IDs")
    index = alpha.index
    n = len(index)
    forecasts = alpha.to_numpy(dtype=float)
    previous = _aligned_series(prev_weights, index, "prev_weights")
    dollar_adv = _aligned_series(adv, index, "adv")
    if not np.isfinite(previous).all() or not np.isfinite(nav) or nav <= 0:
        raise ValueError("finite current weights and positive NAV required")
    if not isinstance(covariance, pd.DataFrame):
        raise TypeError("covariance must be a labeled DataFrame")
    if (
        not covariance.index.is_unique
        or not covariance.columns.is_unique
        or set(covariance.index) != set(index)
        or set(covariance.columns) != set(index)
    ):
        raise ValueError("covariance security IDs do not match alpha")
    cov = covariance.reindex(index=index, columns=index).to_numpy(dtype=float)
    if not np.isfinite(cov).all() or not np.allclose(cov, cov.T, atol=1e-12, rtol=1e-8):
        raise ValueError("covariance must be finite and symmetric")
    eigenvalues = np.linalg.eigvalsh(cov)
    if eigenvalues.min() < -1e-12 * max(1.0, float(np.abs(eigenvalues).max())):
        raise ValueError("covariance must be positive semidefinite")
    cov = (cov + cov.T) / 2
    if exposure is None:
        supplied = np.empty((n, 0))
    else:
        if (
            not isinstance(exposure, pd.DataFrame)
            or not exposure.index.is_unique
            or not exposure.columns.is_unique
            or set(exposure.index) != set(index)
        ):
            raise ValueError("exposure must be a DataFrame with matching unique security IDs")
        supplied = exposure.reindex(index).to_numpy(dtype=float)
        if not np.isfinite(supplied).all():
            raise ValueError("exposure must be finite")
    all_exposure = np.column_stack([np.ones(n), supplied])
    _, rmat, pivots = qr(all_exposure, mode="economic", pivoting=True)
    rank = np.linalg.matrix_rank(rmat)
    independent = all_exposure[:, pivots[:rank]].T

    def unchanged(status, reason, **extra):
        current_exposure = all_exposure.T @ previous
        violations = []
        if np.abs(previous).sum() > limits.max_gross + limits.tolerance:
            violations.append("gross")
        if np.abs(previous).max(initial=0) > limits.max_name + limits.tolerance:
            violations.append("single_name")
        if np.abs(current_exposure).max(initial=0) > limits.tolerance:
            violations.append("neutrality")
        return PortfolioResult(
            pd.Series(previous.copy(), index=index, name="target_weight"),
            {
                "reason": reason,
                "execution_allowed": False,
                "assumed_fills": 0,
                "turnover": 0.0,
                "current_constraint_violations": violations,
                **extra,
            },
            status,
        )

    if not np.isfinite(forecasts).all():
        return unchanged("BLOCKED", "NONFINITE_ALPHA")
    if not np.isfinite(dollar_adv).all() or (dollar_adv < 0).any():
        return unchanged("BLOCKED", "MISSING_OR_INVALID_LIQUIDITY")
    if not np.any(forecasts):
        return unchanged("REJECTED", "NO_ALPHA")
    current_exposure = independent @ previous
    current_neutral = np.abs(current_exposure).max(initial=0) <= limits.tolerance

    diagonal = np.diag(cov)
    ridge = max(float(np.nanmedian(diagonal)) * 1e-6, 1e-12)
    system = cov + np.eye(n) * ridge
    try:
        precision_alpha = np.linalg.solve(system, forecasts)
        precision_exposure = np.linalg.solve(system, independent.T)
    except np.linalg.LinAlgError:
        return unchanged("BLOCKED", "COVARIANCE_SOLVE_FAILED")
    middle = independent @ precision_exposure
    correction = precision_exposure @ np.linalg.pinv(middle, rcond=1e-12) @ (
        independent @ precision_alpha
    )
    direction = (precision_alpha - correction) / max(2 * limits.risk_aversion, 1e-12)
    if not np.isfinite(direction).all() or np.abs(direction).sum() <= 1e-12:
        return unchanged("REJECTED", "NO_NEUTRAL_ALPHA_DIRECTION")
    # Normalize the unconstrained solution only when a hard portfolio cap binds.
    scale = 1.0
    gross = np.abs(direction).sum()
    largest = np.abs(direction).max(initial=0)
    if gross > limits.max_gross:
        scale = min(scale, limits.max_gross / gross)
    if largest > limits.max_name:
        scale = min(scale, limits.max_name / largest)
    desired = direction * scale
    if np.abs(independent @ desired).max(initial=0) > limits.tolerance * 10:
        return unchanged("BLOCKED", "NEUTRAL_PROJECTION_FAILED")

    move = desired - previous
    max_step = 1.0
    turnover = np.abs(move).sum()
    if turnover > 0:
        max_step = min(max_step, limits.max_turnover / turnover)
    per_name_capacity = dollar_adv * limits.max_participation / nav
    required = np.abs(move)
    capacity_ratio = np.divide(
        per_name_capacity,
        required,
        out=np.full(n, np.inf),
        where=required > limits.tolerance,
    )
    max_step = min(max_step, float(capacity_ratio.min(initial=1.0)))
    if not current_neutral:
        # A partial move from a non-neutral book remains non-neutral. Require a full,
        # feasible correction rather than pretending that neutrality was achieved.
        if max_step < 1 - limits.tolerance:
            return unchanged("BLOCKED", "CANNOT_NEUTRALIZE_WITHIN_EXECUTION_LIMITS")
        minimum_step = 1.0
    else:
        minimum_step = 0.0

    volatility = np.sqrt(np.maximum(diagonal, 0))
    best = None
    for step in np.linspace(minimum_step, max(0.0, min(1.0, max_step)), 101):
        target = previous + step * move
        delta = np.abs(target - previous)
        try:
            projected = estimate_costs(previous, target, volatility, dollar_adv, nav, costs)
        except ValueError:
            continue
        remaining = 1 - projected["total"]
        if remaining <= 0:
            continue
        if np.abs(target).sum() / remaining > limits.max_gross + limits.tolerance:
            continue
        if np.abs(target).max(initial=0) / remaining > limits.max_name + limits.tolerance:
            continue
        if delta.sum() > limits.max_turnover + limits.tolerance:
            continue
        participation = np.divide(
            delta * nav, dollar_adv, out=np.zeros(n), where=dollar_adv > 0
        )
        if participation.max(initial=0) > limits.max_participation + limits.tolerance:
            continue
        exposure_residual = np.abs(all_exposure.T @ target).max(initial=0)
        if exposure_residual / remaining > limits.tolerance:
            continue
        expected = float(forecasts @ target)
        variance = float(target @ cov @ target)
        objective = expected - limits.risk_aversion * variance - projected["total"]
        candidate = (objective, target, projected, expected, variance, participation)
        if best is None or candidate[0] > best[0]:
            best = candidate
    if best is None:
        return unchanged("BLOCKED", "NO_FEASIBLE_COSTED_TARGET")
    objective, target, projected, expected, variance, participation = best
    delta = np.abs(target - previous)
    if delta.sum() <= limits.tolerance:
        return unchanged("REJECTED", "NO_ECONOMIC_TRADE")
    remaining = 1 - projected["total"]
    metrics = {
        "reason": "SCALABLE_COVARIANCE_AWARE_TARGET",
        "execution_allowed": True,
        "assumed_fills": 0,
        "turnover": float(delta.sum()),
        "gross": float(np.abs(target).sum()),
        "net": float(target.sum()),
        "max_name": float(np.abs(target).max(initial=0)),
        "max_exposure_residual": float(np.abs(all_exposure.T @ target).max(initial=0)),
        "post_cost_gross": float(np.abs(target).sum() / remaining),
        "post_cost_max_name": float(np.abs(target).max(initial=0) / remaining),
        "expected_return": expected,
        "variance": variance,
        "expected_net_return": expected - projected["total"],
        "objective": objective,
        "conservative_cost_reserve": projected["total"],
        "projected_costs": projected,
        "calendar_days": costs.calendar_days,
        "max_participation": float(participation.max(initial=0)),
        "solver_message": "closed-form neutral mean-variance direction plus exact-cost line search",
    }
    return PortfolioResult(
        pd.Series(target, index=index, name="target_weight"), metrics, "ACCEPTED"
    )

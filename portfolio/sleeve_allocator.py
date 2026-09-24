"""Walk-forward strategy-sleeve allocator inspired by Rodrigues and Carrano (2023)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize


@dataclass(frozen=True)
class SleeveAllocatorConfig:
    fit_window: int = 126
    hold_window: int = 63
    objective_mode: str = "paper"
    return_floor: str = "zero"
    max_sleeve: float = 0.60
    max_reallocation_turnover: float = 0.60
    allocation_cost_bps: float = 5.0
    variance_penalty: float = 0.0
    downside_penalty: float = 0.0
    shrinkage_penalty: float = 0.0
    tolerance: float = 1e-7
    max_iterations: int = 1000

    def __post_init__(self):
        if self.objective_mode not in {"equal", "paper", "variance", "downside"}:
            raise ValueError("unknown sleeve allocation objective")
        if self.return_floor not in {"none", "spy", "zero", "equal"}:
            raise ValueError("unknown return floor")
        if not isinstance(self.fit_window, int) or self.fit_window < 20:
            raise ValueError("fit window must contain at least 20 sessions")
        if not isinstance(self.hold_window, int) or self.hold_window < 1:
            raise ValueError("hold window must be positive")
        numeric = (
            self.max_sleeve,
            self.max_reallocation_turnover,
            self.allocation_cost_bps,
            self.variance_penalty,
            self.downside_penalty,
            self.shrinkage_penalty,
            self.tolerance,
            self.max_iterations,
        )
        if not np.isfinite(numeric).all() or min(numeric) < 0:
            raise ValueError("allocator settings must be finite and nonnegative")
        if self.max_sleeve <= 0 or self.max_reallocation_turnover <= 0:
            raise ValueError("allocation caps must be positive")
        if self.tolerance <= 0 or self.max_iterations < 1:
            raise ValueError("positive tolerance and iteration budget required")


@dataclass(frozen=True)
class SleeveAllocationResult:
    daily: pd.DataFrame
    weights: pd.DataFrame
    refits: pd.DataFrame
    status: str
    reason: str
    evaluation_start: pd.Timestamp | None


class SleeveAllocationError(RuntimeError):
    """The frozen allocation problem is infeasible or numerically invalid."""


def _correlation(portfolio: np.ndarray, benchmark: np.ndarray) -> float:
    centered_portfolio = portfolio - portfolio.mean()
    centered_benchmark = benchmark - benchmark.mean()
    denominator = np.linalg.norm(centered_portfolio) * np.linalg.norm(centered_benchmark)
    if denominator <= 1e-14:
        return np.nan
    return float(centered_portfolio @ centered_benchmark / denominator)


def _floor_value(
    mode: str, sleeve_returns: np.ndarray, benchmark: np.ndarray, equal: np.ndarray
) -> float:
    if mode == "none":
        return -np.inf
    if mode == "spy":
        return float(benchmark.mean())
    if mode == "zero":
        return 0.0
    if mode == "equal":
        return float((sleeve_returns @ equal).mean())
    raise ValueError("unknown return floor")


def solve_sleeve_allocation(
    sleeve_returns: pd.DataFrame,
    benchmark_returns: pd.Series,
    previous: pd.Series,
    config: SleeveAllocatorConfig,
    *,
    initial_funding: bool = False,
) -> tuple[pd.Series, dict]:
    """Fit one causal strategy-sleeve allocation target."""
    if not sleeve_returns.index.equals(benchmark_returns.index):
        raise ValueError("sleeve and benchmark fit sessions must align")
    if not previous.index.equals(sleeve_returns.columns):
        raise ValueError("previous allocation must share sleeve ordering")
    if len(sleeve_returns) != config.fit_window:
        raise ValueError("fit data must match the frozen fit window")
    matrix = sleeve_returns.to_numpy(dtype=float)
    benchmark = benchmark_returns.to_numpy(dtype=float)
    if not np.isfinite(matrix).all() or not np.isfinite(benchmark).all():
        raise SleeveAllocationError("NONFINITE_FIT_RETURNS")
    n = matrix.shape[1]
    if n < 2 or config.max_sleeve * n < 1 - config.tolerance:
        raise SleeveAllocationError("SLEEVE_CAP_CANNOT_REACH_FULL_ALLOCATION")
    equal = np.full(n, 1 / n)
    prior = previous.to_numpy(dtype=float)
    if initial_funding:
        prior_for_start = equal
    else:
        prior_for_start = prior

    floor = _floor_value(config.return_floor, matrix, benchmark, equal)
    equal_portfolio = matrix @ equal
    equal_variance = max(float(np.var(equal_portfolio, ddof=1)), 1e-16)
    equal_downside = max(float(np.mean(np.minimum(equal_portfolio, 0) ** 2)), 1e-16)

    def components(weights: np.ndarray) -> dict[str, float]:
        portfolio = matrix @ weights
        corr = _correlation(portfolio, benchmark)
        if not np.isfinite(corr):
            corr = 1e6
        return {
            "absolute_correlation": abs(corr),
            "variance_ratio": float(np.var(portfolio, ddof=1) / equal_variance),
            "downside_ratio": float(np.mean(np.minimum(portfolio, 0) ** 2) / equal_downside),
            "shrinkage": float(np.square(weights - equal).sum()),
            "fit_mean_return": float(portfolio.mean()),
        }

    def objective(weights: np.ndarray) -> float:
        values = components(weights)
        return float(
            values["absolute_correlation"]
            + config.variance_penalty * values["variance_ratio"]
            + config.downside_penalty * values["downside_ratio"]
            + config.shrinkage_penalty * values["shrinkage"]
        )

    if config.objective_mode == "equal":
        target = equal
        values = components(target)
        if values["fit_mean_return"] < floor - config.tolerance:
            raise SleeveAllocationError("EQUAL_WEIGHT_RETURN_FLOOR_INFEASIBLE")
    else:
        constraints = [
            {"type": "eq", "fun": lambda weights: weights.sum() - 1},
            {"type": "ineq", "fun": lambda weights: (matrix @ weights).mean() - floor},
        ]
        if not initial_funding:
            constraints.append(
                {
                    "type": "ineq",
                    "fun": lambda weights: (
                        config.max_reallocation_turnover - np.abs(weights - prior).sum()
                    ),
                }
            )
        starts = [equal, prior_for_start]
        for sleeve in range(min(3, n)):
            start = np.full(n, 0.40 / (n - 1))
            start[sleeve] = 0.60
            starts.append(start)
        solutions = []
        for start in starts:
            result = minimize(
                objective,
                start,
                method="SLSQP",
                bounds=[(0.0, config.max_sleeve)] * n,
                constraints=constraints,
                options={"ftol": 1e-12, "maxiter": config.max_iterations, "disp": False},
            )
            candidate = np.asarray(result.x, dtype=float)
            mean_return = float((matrix @ candidate).mean())
            turnover = float(np.abs(candidate - prior).sum())
            feasible = (
                np.isfinite(candidate).all()
                and abs(candidate.sum() - 1) <= config.tolerance
                and candidate.min() >= -config.tolerance
                and candidate.max() <= config.max_sleeve + config.tolerance
                and mean_return >= floor - config.tolerance
                and (
                    initial_funding
                    or turnover <= config.max_reallocation_turnover + config.tolerance
                )
            )
            if result.success and feasible:
                solutions.append((objective(candidate), candidate, result.message))
        if not solutions:
            raise SleeveAllocationError("NO_FEASIBLE_ALLOCATION_SOLUTION")
        _, target, message = min(solutions, key=lambda item: item[0])
        values = components(target)
        values["solver_message"] = str(message)

    turnover = float(np.abs(target - prior).sum())
    metrics = {
        **values,
        "return_floor": floor,
        "turnover": turnover,
        "objective": objective(target),
        "maximum_weight": float(target.max()),
        "minimum_weight": float(target.min()),
    }
    return pd.Series(target, index=previous.index, name="target_weight"), metrics


def run_sleeve_allocation(
    sleeve_returns: pd.DataFrame,
    benchmark_returns: pd.Series,
    config: SleeveAllocatorConfig,
    *,
    common_warmup: int = 126,
    implementation_delay_sessions: int = 0,
) -> SleeveAllocationResult:
    """Run the frozen close-to-next-session walk-forward allocation."""
    if not sleeve_returns.index.equals(benchmark_returns.index):
        raise ValueError("sleeve and benchmark sessions must align")
    if sleeve_returns.index.has_duplicates or not sleeve_returns.index.is_monotonic_increasing:
        raise ValueError("sessions must be unique and increasing")
    if len(sleeve_returns) <= common_warmup or config.fit_window > common_warmup:
        raise ValueError("insufficient history for common warm-up")
    if not isinstance(implementation_delay_sessions, int) or implementation_delay_sessions < 0:
        raise ValueError("implementation delay must be a nonnegative integer")
    if sleeve_returns.columns.has_duplicates or len(sleeve_returns.columns) < 2:
        raise ValueError("at least two uniquely named sleeves are required")
    if not np.isfinite(sleeve_returns.to_numpy(dtype=float)).all():
        raise ValueError("sleeve returns must be finite")
    if not np.isfinite(benchmark_returns.to_numpy(dtype=float)).all():
        raise ValueError("benchmark returns must be finite")

    columns = sleeve_returns.columns
    active = pd.Series(0.0, index=columns)
    pending_cost = 0.0
    records: list[dict] = []
    weights: list[pd.Series] = []
    refits: list[dict] = []
    queued: dict | None = None
    first_decision = common_warmup - 1
    evaluation_start = sleeve_returns.index[common_warmup]

    for offset, session in enumerate(sleeve_returns.index):
        realized = sleeve_returns.loc[session]
        sleeve_pnl = float((active * realized).sum())
        net_return = sleeve_pnl - pending_cost
        records.append(
            {
                "session": session,
                "sleeve_return": sleeve_pnl,
                "allocation_cost": pending_cost,
                "net_return": net_return,
                "gross_allocation": float(active.sum()),
            }
        )
        weights.append(active.rename(session))
        if active.sum() > 0:
            denominator = 1 + sleeve_pnl
            if denominator <= 0 or not np.isfinite(denominator):
                return SleeveAllocationResult(
                    pd.DataFrame(records).set_index("session"), pd.DataFrame(weights),
                    pd.DataFrame(refits), "INVALID", "NONPOSITIVE_NAV", evaluation_start
                )
            active = active.mul(1 + realized).div(denominator)
        pending_cost = 0.0

        if queued is not None and offset == queued["execute_offset"]:
            target = queued["target"]
            execution_turnover = float((target - active).abs().sum())
            initial = active.sum() <= config.tolerance
            if (
                not initial
                and execution_turnover
                > config.max_reallocation_turnover + config.tolerance
            ):
                return SleeveAllocationResult(
                    pd.DataFrame(records).set_index("session"), pd.DataFrame(weights),
                    pd.DataFrame(refits), "INVALID", "DELAYED_TURNOVER_VIOLATION",
                    evaluation_start,
                )
            pending_cost = execution_turnover * config.allocation_cost_bps / 10_000
            active = target
            metrics = queued["metrics"]
            refits.append(
                {
                    "session": session,
                    "decision_session": queued["decision_session"],
                    "fit_start": queued["fit_start"],
                    "fit_end": queued["fit_end"],
                    **metrics,
                    "turnover": execution_turnover,
                    **{f"weight_{name}": target[name] for name in columns},
                }
            )
            queued = None

        scheduled = offset >= first_decision and (
            (offset - first_decision) % config.hold_window == 0
        )
        if not scheduled:
            continue
        fit_start = offset - config.fit_window + 1
        if fit_start < 0:
            continue
        fit_returns = sleeve_returns.iloc[fit_start : offset + 1]
        fit_benchmark = benchmark_returns.iloc[fit_start : offset + 1]
        initial = active.sum() <= config.tolerance
        try:
            target, metrics = solve_sleeve_allocation(
                fit_returns, fit_benchmark, active, config, initial_funding=initial
            )
        except (SleeveAllocationError, ValueError) as exc:
            return SleeveAllocationResult(
                pd.DataFrame(records).set_index("session"), pd.DataFrame(weights),
                pd.DataFrame(refits), "INVALID", str(exc), evaluation_start
            )
        if implementation_delay_sessions:
            if queued is not None:
                return SleeveAllocationResult(
                    pd.DataFrame(records).set_index("session"), pd.DataFrame(weights),
                    pd.DataFrame(refits), "INVALID", "OVERLAPPING_DELAYED_TARGETS",
                    evaluation_start,
                )
            queued = {
                "execute_offset": offset + implementation_delay_sessions,
                "decision_session": session,
                "fit_start": fit_returns.index[0],
                "fit_end": fit_returns.index[-1],
                "target": target,
                "metrics": metrics,
            }
        else:
            pending_cost = metrics["turnover"] * config.allocation_cost_bps / 10_000
            active = target
            refits.append(
                {
                    "session": session,
                    "decision_session": session,
                    "fit_start": fit_returns.index[0],
                    "fit_end": fit_returns.index[-1],
                    **metrics,
                    **{f"weight_{name}": target[name] for name in columns},
                }
            )

    return SleeveAllocationResult(
        pd.DataFrame(records).set_index("session"),
        pd.DataFrame(weights),
        pd.DataFrame(refits).set_index("session"),
        "COMPLETED", "OK", evaluation_start
    )

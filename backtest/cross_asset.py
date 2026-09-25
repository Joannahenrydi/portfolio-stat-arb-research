"""Cross-asset factor-neutral, covariance-aware portfolio backtest."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from portfolio.optimizer import PortfolioCosts, estimate_costs


@dataclass(frozen=True)
class CrossAssetConfig:
    rebalance_every: int = 5
    covariance_window: int = 126
    covariance_shrinkage: float = 0.50
    max_gross: float = 1.0
    max_name: float = 0.15
    max_turnover: float = 0.25
    max_participation: float = 0.001
    target_buffer: float = 0.95
    risk_aversion: float = 5.0
    nav: float = 100_000.0
    neutrality_tolerance: float = 1e-8


@dataclass(frozen=True)
class CrossAssetResult:
    daily: pd.DataFrame
    weights: pd.DataFrame
    rebalances: pd.DataFrame
    status: str
    reason: str


def _neutral_direction(alpha: np.ndarray, covariance: np.ndarray, exposure: np.ndarray) -> np.ndarray:
    ridge = max(float(np.median(np.diag(covariance))) * 1e-6, 1e-12)
    system = covariance + np.eye(len(alpha)) * ridge
    precision_alpha = np.linalg.solve(system, alpha)
    precision_exposure = np.linalg.solve(system, exposure)
    correction = precision_exposure @ np.linalg.pinv(
        exposure.T @ precision_exposure, rcond=1e-12
    ) @ (exposure.T @ precision_alpha)
    return precision_alpha - correction


def run_cross_asset_backtest(
    expected_return: pd.DataFrame,
    returns: pd.DataFrame,
    adv: pd.DataFrame,
    eligibility: pd.DataFrame,
    factor_loadings: pd.DataFrame,
    *,
    config: CrossAssetConfig | None = None,
    costs: PortfolioCosts | None = None,
) -> CrossAssetResult:
    config = config or CrossAssetConfig()
    costs = costs or PortfolioCosts(
        commission_bps=0, half_spread_bps=1, slippage_bps=1, borrow_annual=0.01,
        calendar_days=7,
    )
    columns = returns.columns
    for frame in (expected_return, adv, eligibility):
        if not frame.index.equals(returns.index) or not frame.columns.equals(columns):
            raise ValueError("wide cross-asset inputs must align")
    if not factor_loadings.index.equals(columns):
        raise ValueError("factor loadings must follow asset columns")
    active = pd.Series(0.0, index=columns)
    pending_cost = pending_turnover = 0.0
    records, weights, rebalances = [], [], []
    last_session = None
    for offset, session in enumerate(returns.index):
        realized = returns.loc[session]
        if (active.ne(0) & realized.isna()).any():
            return CrossAssetResult(
                pd.DataFrame(records).set_index("session"), pd.DataFrame(weights),
                pd.DataFrame(rebalances), "INVALID", "MISSING_HELD_RETURN",
            )
        gross_return = float((active * realized.fillna(0)).sum())
        days = 1 if last_session is None else max(1, (session - last_session).days)
        borrow = (
            float(active.clip(upper=0).abs().sum()) * costs.borrow_annual * days / 365
            * costs.multiplier
        )
        records.append(
            {"session": session, "gross_return": gross_return,
             "transaction_cost": pending_cost, "borrow_cost": borrow,
             "net_return": gross_return - pending_cost - borrow,
             "gross": float(active.abs().sum()), "net": float(active.sum()),
             "turnover": pending_turnover}
        )
        weights.append(active.rename(session))
        denominator = 1 + records[-1]["net_return"]
        if denominator <= 0 or not np.isfinite(denominator):
            return CrossAssetResult(
                pd.DataFrame(records).set_index("session"), pd.DataFrame(weights),
                pd.DataFrame(rebalances), "INVALID", "NONPOSITIVE_NAV",
            )
        active = active.mul(1 + realized.fillna(0)).div(denominator)
        pending_cost = pending_turnover = 0.0
        if offset % config.rebalance_every == 0 and offset >= config.covariance_window:
            usable = (
                eligibility.loc[session].astype(bool)
                & expected_return.loc[session].notna()
                & adv.loc[session].gt(0)
                & factor_loadings.notna().all(axis=1)
            )
            names = columns[usable]
            if len(names) <= factor_loadings.shape[1] + 1:
                if active.abs().sum() <= config.neutrality_tolerance:
                    last_session = session
                    continue
                return CrossAssetResult(
                    pd.DataFrame(records).set_index("session"), pd.DataFrame(weights),
                    pd.DataFrame(rebalances), "INVALID", "INSUFFICIENT_ASSETS_FOR_FACTORS",
                )
            history = returns.loc[:session, names].tail(config.covariance_window)
            covariance = history.cov().to_numpy(dtype=float) * config.rebalance_every
            diagonal = np.diag(np.diag(covariance))
            covariance = (
                (1 - config.covariance_shrinkage) * covariance
                + config.covariance_shrinkage * diagonal
            )
            exposure = np.column_stack(
                [np.ones(len(names)), factor_loadings.loc[names].to_numpy(dtype=float)]
            )
            try:
                direction = _neutral_direction(
                    expected_return.loc[session, names].to_numpy(dtype=float),
                    covariance, exposure,
                )
            except np.linalg.LinAlgError:
                return CrossAssetResult(
                    pd.DataFrame(records).set_index("session"), pd.DataFrame(weights),
                    pd.DataFrame(rebalances), "INVALID", "COVARIANCE_SOLVE_FAILED",
                )
            if not np.isfinite(direction).all() or np.abs(direction).sum() <= 1e-12:
                return CrossAssetResult(
                    pd.DataFrame(records).set_index("session"), pd.DataFrame(weights),
                    pd.DataFrame(rebalances), "INVALID", "NO_NEUTRAL_DIRECTION",
                )
            desired = direction * (
                config.max_gross * config.target_buffer / np.abs(direction).sum()
            )
            cap = config.max_name * config.target_buffer
            if np.abs(desired).max() > cap:
                desired *= cap / np.abs(desired).max()
            basis = exposure
            current_inside = active.loc[names].to_numpy(dtype=float)
            base_inside = current_inside - basis @ np.linalg.lstsq(
                basis, current_inside, rcond=None
            )[0]
            base = pd.Series(0.0, index=columns)
            base.loc[names] = base_inside
            desired_full = pd.Series(0.0, index=columns)
            desired_full.loc[names] = desired
            correction_turnover = float((base - active).abs().sum())
            if correction_turnover > config.max_turnover + config.neutrality_tolerance:
                return CrossAssetResult(
                    pd.DataFrame(records).set_index("session"), pd.DataFrame(weights),
                    pd.DataFrame(rebalances), "INVALID", "NEUTRAL_CORRECTION_INFEASIBLE",
                )
            move = desired_full - base
            capacity = adv.loc[session].fillna(0) * config.max_participation / config.nav
            best = None
            daily_vol = np.sqrt(np.maximum(np.diag(covariance) / config.rebalance_every, 0))
            vol_full = pd.Series(0.0, index=columns)
            vol_full.loc[names] = daily_vol
            for step in np.linspace(0, 1, 101):
                target = base + step * move
                trade = target - active
                turnover = float(trade.abs().sum())
                if turnover > config.max_turnover + config.neutrality_tolerance:
                    continue
                if (trade.abs() > capacity + config.neutrality_tolerance).any():
                    continue
                projected = estimate_costs(
                    active.to_numpy(), target.to_numpy(), vol_full.to_numpy(),
                    adv.loc[session].fillna(0).to_numpy(), config.nav, costs,
                )
                expected = float(
                    expected_return.loc[session].fillna(0).to_numpy() @ target.to_numpy()
                )
                variance = float(
                    target.loc[names].to_numpy() @ covariance @ target.loc[names].to_numpy()
                )
                objective = expected - config.risk_aversion * variance - projected["total"]
                candidate = (objective, target, projected, expected, variance, turnover)
                if best is None or candidate[0] > best[0]:
                    best = candidate
            if best is None:
                return CrossAssetResult(
                    pd.DataFrame(records).set_index("session"), pd.DataFrame(weights),
                    pd.DataFrame(rebalances), "INVALID", "NO_FEASIBLE_TARGET",
                )
            objective, target, projected, expected, variance, turnover = best
            pending_cost = projected["transaction"]
            pending_turnover = turnover
            active = target
            factor_exposure = factor_loadings.T @ target
            rebalances.append(
                {"session": session, "eligible_assets": len(names),
                 "gross": float(target.abs().sum()), "net": float(target.sum()),
                 "turnover": turnover, "expected_return": expected,
                 "variance": variance, "objective": objective,
                 "max_factor_exposure": float(factor_exposure.abs().max()),
                 **{f"exposure_{name}": float(value)
                    for name, value in factor_exposure.items()}}
            )
        last_session = session
    return CrossAssetResult(
        pd.DataFrame(records).set_index("session"), pd.DataFrame(weights),
        pd.DataFrame(rebalances).set_index("session"), "COMPLETED", "OK",
    )

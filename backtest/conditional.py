"""Strict-neutral conditional portfolio backtest used by frozen v5 experiments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.linalg import qr

from portfolio.optimizer import PortfolioCosts, estimate_costs


@dataclass(frozen=True)
class ConditionalConfig:
    rebalance_every: int = 5
    volatility_window: int = 60
    adv_window: int = 60
    max_gross: float = 1.0
    max_name: float = 0.01
    max_turnover: float = 0.25
    max_participation: float = 0.001
    target_buffer: float = 0.95
    nav: float = 100_000.0
    residual_vol_mode: str = "none"
    short_multiplier: float = 1.0
    regime_mode: str = "none"
    neutrality_tolerance: float = 1e-8

    def __post_init__(self):
        if self.residual_vol_mode not in {"none", "mild", "strong", "capped"}:
            raise ValueError("unknown residual-volatility mode")
        if self.regime_mode not in {"none", "mild", "defensive"}:
            raise ValueError("unknown regime mode")
        if not 0 < self.short_multiplier <= 1:
            raise ValueError("short multiplier must lie in (0, 1]")
        positive = (
            self.rebalance_every,
            self.volatility_window,
            self.adv_window,
            self.max_gross,
            self.max_name,
            self.max_turnover,
            self.max_participation,
            self.target_buffer,
            self.nav,
            self.neutrality_tolerance,
        )
        if not np.isfinite(positive).all() or min(positive) <= 0:
            raise ValueError("conditional portfolio settings must be finite and positive")
        if self.target_buffer > 1:
            raise ValueError("target buffer cannot exceed one")


@dataclass(frozen=True)
class ConditioningArtifact:
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    sorted_dispersion: tuple[float, ...]


@dataclass(frozen=True)
class ConditionalBacktestResult:
    daily: pd.DataFrame
    weights: pd.DataFrame
    rebalances: pd.DataFrame
    status: str
    reason: str


class StrictNeutralityError(RuntimeError):
    """An executable strictly neutral target does not exist."""


def fit_conditioning_artifact(
    prior_dispersion: pd.Series, train_start, train_end
) -> ConditioningArtifact:
    start, end = pd.Timestamp(train_start), pd.Timestamp(train_end)
    values = prior_dispersion.loc[start:end].dropna().to_numpy(dtype=float)
    if len(values) < 252 or not np.isfinite(values).all():
        raise ValueError("at least 252 finite training dispersion observations are required")
    return ConditioningArtifact(start, end, tuple(np.sort(values)))


def _dispersion_percentile(value: float, artifact: ConditioningArtifact) -> float:
    reference = np.asarray(artifact.sorted_dispersion)
    if not np.isfinite(value):
        return np.nan
    return float(np.searchsorted(reference, value, side="right") / len(reference))


def regime_gross_scalar(
    mode: str,
    prior_dispersion: float,
    prior_drawdown: float,
    artifact: ConditioningArtifact,
) -> float:
    if mode == "none":
        return 1.0
    percentile = _dispersion_percentile(prior_dispersion, artifact)
    if not np.isfinite(percentile) or not np.isfinite(prior_drawdown):
        return 0.0
    if mode == "mild":
        dispersion_scale = 0.80 + 0.30 * percentile
        normal, moderate, stress, floor, ceiling = 0.80, 1.00, 0.40, 0.35, 1.00
    elif mode == "defensive":
        dispersion_scale = 0.70 + 0.30 * percentile
        normal, moderate, stress, floor, ceiling = 0.70, 0.90, 0.25, 0.25, 0.90
    else:
        raise ValueError("unknown regime mode")
    if prior_drawdown >= -0.05:
        drawdown_scale = normal
    elif prior_drawdown >= -0.10:
        fraction = (-prior_drawdown - 0.05) / 0.05
        drawdown_scale = normal + fraction * (moderate - normal)
    elif prior_drawdown >= -0.15:
        fraction = (-prior_drawdown - 0.10) / 0.05
        drawdown_scale = moderate + fraction * (stress - moderate)
    else:
        drawdown_scale = stress
    return float(np.clip(dispersion_scale * drawdown_scale, floor, ceiling))


def residual_volatility_multiplier(percentile: pd.Series, mode: str) -> pd.Series:
    if mode == "none":
        return pd.Series(1.0, index=percentile.index)
    if mode == "mild":
        return 0.75 + 0.50 * percentile
    if mode == "strong":
        return 0.50 + percentile
    if mode == "capped":
        return (0.60 + 0.80 * percentile).clip(0.70, 1.30)
    raise ValueError("unknown residual-volatility mode")


def _neutral_basis(beta: pd.Series, sectors: pd.Series) -> np.ndarray:
    dummies = pd.get_dummies(sectors, dtype=float, drop_first=False)
    exposure = np.column_stack([np.ones(len(beta)), beta.to_numpy(dtype=float), dummies])
    _, rmat, pivots = qr(exposure, mode="economic", pivoting=True)
    rank = np.linalg.matrix_rank(rmat)
    return exposure[:, pivots[:rank]]


def _project(vector: np.ndarray, basis: np.ndarray) -> np.ndarray:
    return vector - basis @ np.linalg.lstsq(basis, vector, rcond=None)[0]


def strict_neutral_target(
    score: pd.Series,
    beta: pd.Series,
    sectors: pd.Series,
    previous: pd.Series,
    adv: pd.Series,
    residual_volatility: pd.Series,
    gross_scalar: float,
    config: ConditionalConfig,
) -> tuple[pd.Series, dict]:
    """Return an executable target whose post-trade exposures are exactly neutral."""
    index = score.index
    for name, series in (
        ("beta", beta),
        ("sectors", sectors),
        ("previous", previous),
        ("adv", adv),
        ("residual_volatility", residual_volatility),
    ):
        if not series.index.equals(index):
            raise ValueError(f"{name} must share score index and ordering")
    if not np.isfinite(gross_scalar) or not 0 <= gross_scalar <= 1:
        raise ValueError("gross scalar must lie in [0, 1]")
    usable = score.notna() & beta.notna() & sectors.notna() & adv.gt(0) & residual_volatility.notna()
    names = index[usable]
    outside = index[~usable]
    if previous.loc[outside].abs().sum() > config.neutrality_tolerance:
        raise StrictNeutralityError("HELD_NAME_BECAME_INELIGIBLE")
    if len(names) < 50:
        if previous.abs().sum() <= config.neutrality_tolerance:
            return previous.copy(), {"reason": "NO_SIGNAL", "gross_scalar": gross_scalar}
        raise StrictNeutralityError("INSUFFICIENT_ELIGIBLE_NAMES")
    selected_beta = beta.loc[names].astype(float)
    selected_sectors = sectors.loc[names]
    basis = _neutral_basis(selected_beta, selected_sectors)
    ranked = score.loc[names].rank(method="average", pct=True) - 0.5
    ranked.loc[ranked < 0] *= config.short_multiplier
    vol_percentile = residual_volatility.loc[names].rank(method="average", pct=True)
    ranked *= residual_volatility_multiplier(vol_percentile, config.residual_vol_mode)
    desired = _project(ranked.to_numpy(dtype=float), basis)
    gross = np.abs(desired).sum()
    if gross <= config.neutrality_tolerance:
        return previous.copy(), {"reason": "NO_NEUTRAL_SIGNAL", "gross_scalar": gross_scalar}
    desired *= config.max_gross * config.target_buffer * gross_scalar / gross
    largest = np.abs(desired).max(initial=0)
    desired_name_cap = config.max_name * config.target_buffer
    if largest > desired_name_cap:
        desired *= desired_name_cap / largest

    current = previous.loc[names].to_numpy(dtype=float)
    neutral_current = _project(current, basis)
    correction = neutral_current - current
    capacity = adv.loc[names].to_numpy(dtype=float) * config.max_participation / config.nav
    if np.abs(correction).sum() > config.max_turnover + config.neutrality_tolerance:
        raise StrictNeutralityError("NEUTRALITY_CORRECTION_EXCEEDS_TURNOVER")
    if (np.abs(correction) > capacity + config.neutrality_tolerance).any():
        raise StrictNeutralityError("NEUTRALITY_CORRECTION_EXCEEDS_LIQUIDITY")
    direction = desired - neutral_current
    chosen = None
    for step in np.linspace(1.0, 0.0, 1001):
        candidate = neutral_current + step * direction
        trade = candidate - current
        if np.abs(candidate).max(initial=0) > config.max_name + config.neutrality_tolerance:
            continue
        if np.abs(candidate).sum() > config.max_gross + config.neutrality_tolerance:
            continue
        if np.abs(trade).sum() > config.max_turnover + config.neutrality_tolerance:
            continue
        if (np.abs(trade) > capacity + config.neutrality_tolerance).any():
            continue
        chosen = candidate
        break
    if chosen is None:
        raise StrictNeutralityError("NO_EXECUTABLE_STRICT_NEUTRAL_TARGET")
    residual = np.abs(basis.T @ chosen).max(initial=0)
    if residual > config.neutrality_tolerance:
        raise StrictNeutralityError("POST_TRADE_NEUTRALITY_FAILURE")
    target = pd.Series(0.0, index=index, name="target_weight")
    target.loc[names] = chosen
    sector_exposure = target.groupby(sectors).sum().abs().max()
    metrics = {
        "reason": "STRICT_NEUTRAL_TARGET",
        "gross_scalar": gross_scalar,
        "desired_gross": float(np.abs(desired).sum()),
        "actual_gross": float(target.abs().sum()),
        "turnover": float((target - previous).abs().sum()),
        "net_exposure": float(target.sum()),
        "beta_exposure": float((target * beta).sum()),
        "max_sector_exposure": float(sector_exposure),
        "max_linear_exposure": float(residual),
    }
    return target, metrics


def run_conditional_backtest(
    score: pd.DataFrame,
    returns: pd.DataFrame,
    beta: pd.DataFrame,
    sectors: pd.Series,
    raw_close: pd.DataFrame,
    raw_volume: pd.DataFrame,
    residual_volatility: pd.DataFrame,
    prior_dispersion: pd.Series,
    prior_market_drawdown: pd.Series,
    artifact: ConditioningArtifact,
    *,
    config: ConditionalConfig | None = None,
    costs: PortfolioCosts | None = None,
) -> ConditionalBacktestResult:
    config = config or ConditionalConfig()
    costs = costs or PortfolioCosts()
    columns = returns.columns
    for frame in (score, beta, raw_close, raw_volume, residual_volatility):
        if not frame.index.equals(returns.index) or not frame.columns.equals(columns):
            raise ValueError("all wide inputs must share sessions and symbols")
    if not sectors.index.equals(columns):
        raise ValueError("sectors must share symbol ordering")
    for series in (prior_dispersion, prior_market_drawdown):
        if not series.index.equals(returns.index):
            raise ValueError("regime inputs must share sessions")
    volatility = returns.rolling(config.volatility_window, min_periods=config.volatility_window).std()
    adv = (raw_close * raw_volume).rolling(config.adv_window, min_periods=config.adv_window).median()
    active = pd.Series(0.0, index=columns)
    records, weights, rebalances = [], [], []
    pending_cost = 0.0
    pending_turnover = 0.0
    last_session = None
    for offset, session in enumerate(returns.index):
        realized = returns.loc[session]
        missing_held = active.ne(0) & realized.isna()
        if missing_held.any():
            return ConditionalBacktestResult(
                pd.DataFrame(records).set_index("session"),
                pd.DataFrame(weights),
                pd.DataFrame(rebalances),
                "INVALID",
                "MISSING_HELD_RETURN",
            )
        gross_return = float((active * realized.fillna(0)).sum())
        calendar_days = 1 if last_session is None else max(1, (session - last_session).days)
        borrow = (
            float(active.clip(upper=0).abs().sum())
            * costs.borrow_annual
            * calendar_days
            / 365
            * costs.multiplier
        )
        net_return = gross_return - pending_cost - borrow
        records.append(
            {
                "session": session,
                "gross_return": gross_return,
                "transaction_cost": pending_cost,
                "borrow_cost": borrow,
                "net_return": net_return,
                "gross": float(active.abs().sum()),
                "net": float(active.sum()),
                "turnover": pending_turnover,
            }
        )
        weights.append(active.rename(session))
        denominator = 1 + net_return
        if not np.isfinite(denominator) or denominator <= 0:
            return ConditionalBacktestResult(
                pd.DataFrame(records).set_index("session"),
                pd.DataFrame(weights),
                pd.DataFrame(rebalances),
                "INVALID",
                "NONPOSITIVE_NAV",
            )
        active = active.mul(1 + realized.fillna(0)).div(denominator)
        pending_cost = 0.0
        pending_turnover = 0.0
        if offset % config.rebalance_every == 0:
            scalar = regime_gross_scalar(
                config.regime_mode,
                float(prior_dispersion.loc[session]),
                float(prior_market_drawdown.loc[session]),
                artifact,
            )
            try:
                target, target_metrics = strict_neutral_target(
                    score.loc[session],
                    beta.loc[session],
                    sectors,
                    active,
                    adv.loc[session],
                    residual_volatility.loc[session],
                    scalar,
                    config,
                )
                projected = estimate_costs(
                    active.to_numpy(),
                    target.to_numpy(),
                    volatility.loc[session].fillna(0).to_numpy(),
                    adv.loc[session].fillna(0).to_numpy(),
                    config.nav,
                    PortfolioCosts(
                        commission_bps=costs.commission_bps,
                        half_spread_bps=costs.half_spread_bps,
                        slippage_bps=costs.slippage_bps,
                        impact_coefficient=costs.impact_coefficient,
                        borrow_annual=0.0,
                        multiplier=costs.multiplier,
                    ),
                )
            except (StrictNeutralityError, ValueError) as exc:
                return ConditionalBacktestResult(
                    pd.DataFrame(records).set_index("session"),
                    pd.DataFrame(weights),
                    pd.DataFrame(rebalances),
                    "INVALID",
                    str(exc),
                )
            pending_cost = projected["transaction"]
            pending_turnover = float((target - active).abs().sum())
            active = target
            rebalances.append({"session": session, **target_metrics})
        last_session = session
    return ConditionalBacktestResult(
        pd.DataFrame(records).set_index("session"),
        pd.DataFrame(weights),
        pd.DataFrame(rebalances).set_index("session"),
        "COMPLETED",
        "OK",
    )

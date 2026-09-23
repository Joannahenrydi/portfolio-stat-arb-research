"""Frozen v7 short-horizon residual-reversal research engine."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.linalg import qr

from portfolio.optimizer import PortfolioCosts, estimate_costs


@dataclass(frozen=True)
class ResidualReversalConfig:
    rebalance_every: int = 3
    volatility_window: int = 60
    max_gross: float = 1.0
    max_name: float = 0.01
    max_turnover: float = 0.25
    max_participation: float = 0.001
    target_buffer: float = 0.95
    residual_vol_mode: str = "none"
    stress_scaler: bool = False
    separate_side_ranks: bool = False
    nav: float = 100_000.0
    neutrality_tolerance: float = 1e-8

    def __post_init__(self):
        if self.residual_vol_mode not in {"none", "mild", "strong"}:
            raise ValueError("unknown residual-volatility mode")
        integer_fields = (self.rebalance_every, self.volatility_window)
        if any(not isinstance(value, int) or value < 1 for value in integer_fields):
            raise ValueError("window and rebalance settings must be positive integers")
        positive = (
            self.max_gross,
            self.max_name,
            self.max_turnover,
            self.max_participation,
            self.target_buffer,
            self.nav,
            self.neutrality_tolerance,
        )
        if not np.isfinite(positive).all() or min(positive) <= 0:
            raise ValueError("portfolio settings must be finite and positive")
        if self.target_buffer > 1:
            raise ValueError("target buffer cannot exceed one")


@dataclass(frozen=True)
class StressArtifact:
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    sorted_market_volatility: tuple[float, ...]


@dataclass(frozen=True)
class ResidualReversalResult:
    daily: pd.DataFrame
    weights: pd.DataFrame
    rebalances: pd.DataFrame
    status: str
    reason: str


class ResidualReversalError(RuntimeError):
    """A required input or executable strictly neutral target is unavailable."""


def fit_stress_artifact(
    prior_market_volatility: pd.Series, train_start, train_end
) -> StressArtifact:
    start, end = pd.Timestamp(train_start), pd.Timestamp(train_end)
    values = prior_market_volatility.loc[start:end].dropna().to_numpy(dtype=float)
    if len(values) < 252 or not np.isfinite(values).all():
        raise ValueError("at least 252 finite training volatility observations are required")
    return StressArtifact(start, end, tuple(np.sort(values)))


def stress_gross_scalar(value: float, artifact: StressArtifact) -> float:
    """Frozen continuous scaler based on the train empirical volatility CDF."""
    if not np.isfinite(value):
        raise ResidualReversalError("MISSING_STRESS_INPUT")
    reference = np.asarray(artifact.sorted_market_volatility)
    percentile = np.searchsorted(reference, value, side="right") / len(reference)
    if percentile <= 0.80:
        return 1.0
    if percentile <= 0.95:
        return float(1.0 - (percentile - 0.80) / 0.15 * 0.50)
    if percentile <= 0.99:
        return float(0.50 - (percentile - 0.95) / 0.04 * 0.25)
    return 0.25


def cross_sectional_zscore(frame: pd.DataFrame, eligibility: pd.DataFrame) -> pd.DataFrame:
    usable = frame.where(eligibility)
    centered = usable.sub(usable.mean(axis=1), axis=0)
    scale = usable.std(axis=1, ddof=1).where(usable.count(axis=1).ge(2))
    return centered.div(scale.where(scale.gt(0)), axis=0).clip(-3, 3)


def build_v7_scores(
    residual: pd.DataFrame, eligibility: pd.DataFrame
) -> dict[str, pd.DataFrame]:
    """Build the protocol-frozen reference and core residual-reversal forecasts."""
    if not residual.index.equals(eligibility.index) or not residual.columns.equals(
        eligibility.columns
    ):
        raise ValueError("residual and eligibility must align")
    one = cross_sectional_zscore(residual, eligibility)
    three_raw = residual.rolling(3, min_periods=3).sum()
    five_raw = residual.rolling(5, min_periods=5).sum()
    three = cross_sectional_zscore(three_raw, eligibility)
    five = cross_sectional_zscore(five_raw, eligibility)
    return {
        "reference": -three,
        "core": -(0.50 * one + 0.30 * three + 0.20 * five),
    }


def _neutral_basis(beta: pd.Series, sectors: pd.Series) -> np.ndarray:
    dummies = pd.get_dummies(sectors, dtype=float, drop_first=False)
    exposure = np.column_stack([np.ones(len(beta)), beta.to_numpy(dtype=float), dummies])
    _, rmat, pivots = qr(exposure, mode="economic", pivoting=True)
    rank = np.linalg.matrix_rank(rmat)
    return exposure[:, pivots[:rank]]


def _project(vector: np.ndarray, basis: np.ndarray) -> np.ndarray:
    return vector - basis @ np.linalg.lstsq(basis, vector, rcond=None)[0]


def _side_rank(score: pd.Series) -> pd.Series:
    result = score.copy() * np.nan
    positive = score.gt(0)
    negative = score.lt(0)
    if positive.any():
        result.loc[positive] = 0.5 + 0.5 * score.loc[positive].rank(pct=True)
    if negative.any():
        result.loc[negative] = -(0.5 + 0.5 * score.loc[negative].abs().rank(pct=True))
    return result.fillna(0.0)


def residual_volatility_multiplier(percentile: pd.Series, mode: str) -> pd.Series:
    if mode == "none":
        return pd.Series(1.0, index=percentile.index)
    if mode == "mild":
        return 0.75 + 0.50 * percentile
    if mode == "strong":
        return 0.50 + percentile
    raise ValueError("unknown residual-volatility mode")


def strict_liquid_target(
    score: pd.Series,
    beta: pd.Series,
    sectors: pd.Series,
    previous: pd.Series,
    adv: pd.Series,
    residual_volatility: pd.Series,
    eligibility: pd.Series,
    gross_scalar: float,
    config: ResidualReversalConfig,
) -> tuple[pd.Series, dict]:
    """Build a neutral target while explicitly liquidating newly ineligible holdings."""
    index = score.index
    aligned = (beta, sectors, previous, adv, residual_volatility, eligibility)
    if any(not series.index.equals(index) for series in aligned):
        raise ValueError("all target inputs must share symbol ordering")
    if not np.isfinite(gross_scalar) or not 0 < gross_scalar <= 1:
        raise ValueError("gross scalar must lie in (0, 1]")
    usable = (
        eligibility.astype(bool)
        & score.notna()
        & beta.notna()
        & sectors.notna()
        & adv.gt(0)
        & residual_volatility.notna()
    )
    names = index[usable]
    if len(names) < 50:
        if previous.abs().sum() <= config.neutrality_tolerance:
            return previous.copy(), {
                "gross_scalar": gross_scalar,
                "eligible_names": len(names),
                "actual_gross": 0.0,
                "turnover": 0.0,
                "net_exposure": 0.0,
                "beta_exposure": 0.0,
                "max_sector_exposure": 0.0,
                "max_linear_exposure": 0.0,
            }
        raise ResidualReversalError("INSUFFICIENT_ELIGIBLE_NAMES")
    held = previous.ne(0)
    if (held & adv.fillna(0).le(0)).any():
        raise ResidualReversalError("MISSING_LIQUIDITY_FOR_HELD_NAME")

    selected_beta = beta.loc[names].astype(float)
    selected_sectors = sectors.loc[names]
    basis = _neutral_basis(selected_beta, selected_sectors)
    raw_score = score.loc[names].astype(float)
    ranked = _side_rank(raw_score) if config.separate_side_ranks else raw_score.rank(pct=True) - 0.5
    vol_percentile = residual_volatility.loc[names].rank(method="average", pct=True)
    ranked *= residual_volatility_multiplier(vol_percentile, config.residual_vol_mode)
    desired = _project(ranked.to_numpy(dtype=float), basis)
    desired_gross = np.abs(desired).sum()
    if desired_gross <= config.neutrality_tolerance:
        raise ResidualReversalError("NO_NEUTRAL_SIGNAL")
    desired *= config.max_gross * config.target_buffer * gross_scalar / desired_gross
    largest = np.abs(desired).max(initial=0)
    buffered_name_cap = config.max_name * config.target_buffer
    if largest > buffered_name_cap:
        desired *= buffered_name_cap / largest

    current_inside = previous.loc[names].to_numpy(dtype=float)
    neutral_current = _project(current_inside, basis)
    base = pd.Series(0.0, index=index)
    base.loc[names] = neutral_current
    capacity = adv.fillna(0).to_numpy(dtype=float) * config.max_participation / config.nav

    def feasible(candidate: pd.Series) -> bool:
        trade = candidate.to_numpy(dtype=float) - previous.to_numpy(dtype=float)
        return bool(
            candidate.abs().max() <= config.max_name + config.neutrality_tolerance
            and candidate.abs().sum() <= config.max_gross + config.neutrality_tolerance
            and np.abs(trade).sum() <= config.max_turnover + config.neutrality_tolerance
            and (np.abs(trade) <= capacity + config.neutrality_tolerance).all()
        )

    if not feasible(base):
        raise ResidualReversalError("ELIGIBILITY_OR_NEUTRALITY_CORRECTION_INFEASIBLE")
    desired_series = pd.Series(0.0, index=index)
    desired_series.loc[names] = desired
    if feasible(desired_series):
        target = desired_series
    else:
        lower, upper = 0.0, 1.0
        for _ in range(50):
            midpoint = (lower + upper) / 2
            candidate = base + midpoint * (desired_series - base)
            if feasible(candidate):
                lower = midpoint
            else:
                upper = midpoint
        target = base + lower * (desired_series - base)

    residual_exposure = np.abs(basis.T @ target.loc[names].to_numpy()).max(initial=0)
    sector_exposure = target.groupby(sectors).sum().abs().max()
    if residual_exposure > config.neutrality_tolerance:
        raise ResidualReversalError("POST_TRADE_NEUTRALITY_FAILURE")
    metrics = {
        "gross_scalar": gross_scalar,
        "eligible_names": len(names),
        "actual_gross": float(target.abs().sum()),
        "turnover": float((target - previous).abs().sum()),
        "net_exposure": float(target.sum()),
        "beta_exposure": float((target * beta).sum()),
        "max_sector_exposure": float(sector_exposure),
        "max_linear_exposure": float(residual_exposure),
    }
    return target, metrics


def run_residual_reversal_backtest(
    score: pd.DataFrame,
    returns: pd.DataFrame,
    beta: pd.DataFrame,
    sectors: pd.Series,
    raw_close: pd.DataFrame,
    raw_volume: pd.DataFrame,
    adv: pd.DataFrame,
    residual_volatility: pd.DataFrame,
    eligibility: pd.DataFrame,
    prior_market_volatility: pd.Series,
    stress_artifact: StressArtifact,
    *,
    config: ResidualReversalConfig | None = None,
    costs: PortfolioCosts | None = None,
) -> ResidualReversalResult:
    config = config or ResidualReversalConfig()
    costs = costs or PortfolioCosts()
    columns = returns.columns
    wide = (score, beta, raw_close, raw_volume, adv, residual_volatility, eligibility)
    if any(
        not frame.index.equals(returns.index) or not frame.columns.equals(columns)
        for frame in wide
    ):
        raise ValueError("all wide inputs must share sessions and symbols")
    if not sectors.index.equals(columns) or not prior_market_volatility.index.equals(returns.index):
        raise ValueError("sector or market-volatility input is misaligned")

    daily_volatility = returns.rolling(
        config.volatility_window, min_periods=config.volatility_window
    ).std()
    active = pd.Series(0.0, index=columns)
    records: list[dict] = []
    weights: list[pd.Series] = []
    rebalances: list[dict] = []
    pending_cost = pending_turnover = 0.0
    last_session = None
    for offset, session in enumerate(returns.index):
        realized = returns.loc[session]
        if (active.ne(0) & realized.isna()).any():
            return ResidualReversalResult(
                pd.DataFrame(records).set_index("session"), pd.DataFrame(weights),
                pd.DataFrame(rebalances), "INVALID", "MISSING_HELD_RETURN"
            )
        gross_return = float((active * realized.fillna(0)).sum())
        calendar_days = 1 if last_session is None else max(1, (session - last_session).days)
        borrow = (
            float(active.clip(upper=0).abs().sum())
            * costs.borrow_annual * calendar_days / 365 * costs.multiplier
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
            return ResidualReversalResult(
                pd.DataFrame(records).set_index("session"), pd.DataFrame(weights),
                pd.DataFrame(rebalances), "INVALID", "NONPOSITIVE_NAV"
            )
        active = active.mul(1 + realized.fillna(0)).div(denominator)
        pending_cost = pending_turnover = 0.0
        if offset % config.rebalance_every == 0:
            try:
                scalar = (
                    stress_gross_scalar(
                        float(prior_market_volatility.loc[session]), stress_artifact
                    )
                    if config.stress_scaler else 1.0
                )
                target, metrics = strict_liquid_target(
                    score.loc[session], beta.loc[session], sectors, active, adv.loc[session],
                    residual_volatility.loc[session], eligibility.loc[session], scalar, config
                )
                projected = estimate_costs(
                    active.to_numpy(), target.to_numpy(),
                    daily_volatility.loc[session].fillna(0).to_numpy(),
                    adv.loc[session].fillna(0).to_numpy(), config.nav,
                    PortfolioCosts(
                        commission_bps=costs.commission_bps,
                        half_spread_bps=costs.half_spread_bps,
                        slippage_bps=costs.slippage_bps,
                        impact_coefficient=costs.impact_coefficient,
                        borrow_annual=0.0,
                        multiplier=costs.multiplier,
                    ),
                )
            except (ResidualReversalError, ValueError) as exc:
                return ResidualReversalResult(
                    pd.DataFrame(records).set_index("session"), pd.DataFrame(weights),
                    pd.DataFrame(rebalances), "INVALID", str(exc)
                )
            pending_cost = projected["transaction"]
            pending_turnover = float((target - active).abs().sum())
            active = target
            rebalances.append({"session": session, **metrics})
        last_session = session
    return ResidualReversalResult(
        pd.DataFrame(records).set_index("session"),
        pd.DataFrame(weights),
        pd.DataFrame(rebalances).set_index("session"),
        "COMPLETED", "OK"
    )

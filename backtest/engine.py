"""Fast causal cross-sectional equity backtests with explicit neutralization and costs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from portfolio.optimizer import PortfolioCosts, estimate_costs


@dataclass(frozen=True)
class FastPortfolioConfig:
    rebalance_every: int = 5
    beta_window: int = 126
    volatility_window: int = 60
    adv_window: int = 60
    max_gross: float = 1.0
    max_name: float = 0.01
    max_turnover: float = 0.25
    max_participation: float = 0.001
    nav: float = 100_000.0
    score_transform: str = "rank"
    tail_fraction: float = 0.20


def prior_market_beta(returns: pd.DataFrame, market: pd.Series, window: int = 126) -> pd.DataFrame:
    """Rolling beta using observations strictly before each decision session."""
    variance = market.rolling(window, min_periods=window).var().shift(1)
    return returns.rolling(window, min_periods=window).cov(market).shift(1).div(
        variance.where(variance > 0), axis=0
    )


def residual_returns(
    returns: pd.DataFrame, market: pd.Series, sectors: pd.Series, beta_window: int = 126
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Remove prior market beta and contemporaneous leave-one-out sector return."""
    beta = prior_market_beta(returns, market, beta_window)
    market_residual = returns.sub(beta.mul(market, axis=0))
    result = market_residual.copy() * np.nan
    for sector in sorted(sectors.dropna().unique()):
        names = sectors.index[sectors.eq(sector)]
        group = market_residual[names]
        count = group.notna().sum(axis=1)
        peer = group.rsub(group.sum(axis=1, min_count=1), axis=0).div(
            (count - 1).where(count >= 3), axis=0
        )
        result[names] = group - peer
    return result, beta


def candidate_scores(residual: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Return the finite, predeclared signal grid."""
    scores: dict[str, pd.DataFrame] = {}
    for window in (1, 3, 5, 10):
        scores[f"reversal_{window}"] = -residual.rolling(window, min_periods=window).sum()
    for window, skip in ((21, 5), (63, 5), (126, 21), (252, 21)):
        scores[f"momentum_{window}_skip_{skip}"] = residual.shift(skip).rolling(
            window, min_periods=window
        ).sum()
    scores["blend_rev3_mom63"] = (
        scores["reversal_3"].rank(axis=1, pct=True)
        + scores["momentum_63_skip_5"].rank(axis=1, pct=True)
        - 1
    )
    return scores


def _projected_target(
    score: pd.Series,
    beta: pd.Series,
    sectors: pd.Series,
    previous: pd.Series,
    adv: pd.Series,
    config: FastPortfolioConfig,
) -> pd.Series:
    usable = score.notna() & beta.notna() & adv.gt(0)
    names = score.index[usable]
    if len(names) < 50:
        return previous.copy()
    percentile = score[names].rank(method="average", pct=True)
    if config.score_transform == "rank":
        ranked = percentile - 0.5
    elif config.score_transform == "tail":
        if not 0 < config.tail_fraction < 0.5:
            raise ValueError("tail_fraction must be between zero and one half")
        ranked = pd.Series(0.0, index=names)
        ranked.loc[percentile <= config.tail_fraction] = -1.0
        ranked.loc[percentile >= 1 - config.tail_fraction] = 1.0
    else:
        raise ValueError("score_transform must be rank or tail")
    dummies = pd.get_dummies(sectors[names], dtype=float, drop_first=False)
    exposure = np.column_stack([np.ones(len(names)), dummies.to_numpy(), beta[names].to_numpy()])
    independent = np.linalg.qr(exposure)[1]
    rank = np.linalg.matrix_rank(independent)
    _, _, pivots = __import__("scipy.linalg").linalg.qr(exposure, pivoting=True, mode="economic")
    x = exposure[:, pivots[:rank]]
    residual_score = ranked.to_numpy() - x @ np.linalg.lstsq(x, ranked.to_numpy(), rcond=None)[0]
    target = pd.Series(0.0, index=score.index)
    gross = np.abs(residual_score).sum()
    if gross <= 1e-12:
        return previous.copy()
    target.loc[names] = residual_score / gross * config.max_gross
    # Uniform scaling preserves every linear neutrality constraint.
    largest = target.abs().max()
    if largest > config.max_name:
        target *= config.max_name / largest
    capacity = adv * config.max_participation / config.nav
    move = target - previous
    scale = min(1.0, config.max_turnover / max(move.abs().sum(), 1e-12))
    ratios = capacity.div(move.abs().where(move.abs() > 0)).replace([np.inf, -np.inf], np.nan)
    if ratios.notna().any():
        scale = min(scale, float(ratios.min()))
    return previous + scale * move


def run_fast_backtest(
    score: pd.DataFrame,
    returns: pd.DataFrame,
    beta: pd.DataFrame,
    sectors: pd.Series,
    raw_close: pd.DataFrame,
    raw_volume: pd.DataFrame,
    *,
    config: FastPortfolioConfig | None = None,
    costs: PortfolioCosts | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run one signal; decision at close t first applies to return t+1."""
    config = config or FastPortfolioConfig()
    costs = costs or PortfolioCosts()
    columns = returns.columns
    for frame in (score, beta, raw_close, raw_volume):
        if not frame.index.equals(returns.index) or not frame.columns.equals(columns):
            raise ValueError("all wide frames must have identical sessions and symbols")
    if not sectors.index.equals(columns):
        raise ValueError("sector labels must match symbol columns")
    volatility = returns.rolling(config.volatility_window, min_periods=config.volatility_window).std()
    adv = (raw_close * raw_volume).rolling(config.adv_window, min_periods=config.adv_window).median()
    previous = pd.Series(0.0, index=columns)
    active = previous.copy()
    records: list[dict] = []
    weights: list[pd.Series] = []
    pending_cost = 0.0
    pending_turnover = 0.0
    last_session = None
    for offset, session in enumerate(returns.index):
        daily = returns.loc[session]
        gross_return = float((active * daily.fillna(0)).sum())
        missing_held = bool((active.ne(0) & daily.isna()).any())
        if missing_held:
            missing_names = list(columns[active.ne(0) & daily.isna()])
            raise ValueError(
                f"missing return for held securities on {session.date()}: {missing_names[:10]}"
            )
        calendar_days = 1 if last_session is None else max(1, (session - last_session).days)
        borrow = (
            float(active.clip(upper=0).abs().sum())
            * costs.borrow_annual
            * calendar_days
            / 365
            * costs.multiplier
        )
        net_return = gross_return - pending_cost - borrow
        records.append({
            "session": session, "gross_return": gross_return, "transaction_cost": pending_cost,
            "borrow_cost": borrow, "net_return": net_return, "gross": float(active.abs().sum()),
            "net": float(active.sum()), "turnover": pending_turnover,
            "missing_held_return": missing_held,
        })
        weights.append(active.rename(session))
        denominator = 1 + net_return
        if not np.isfinite(denominator) or denominator <= 0:
            raise ValueError(f"portfolio NAV became nonpositive on {session.date()}")
        # Holdings drift with their marks. This prevents an uncharged implicit daily
        # rebalance back to yesterday's target weights.
        active = active.mul(1 + daily.fillna(0)).div(denominator)
        pending_cost = 0.0
        pending_turnover = 0.0
        if offset % config.rebalance_every == 0:
            target = _projected_target(
                score.loc[session], beta.loc[session], sectors, active, adv.loc[session], config
            )
            vol = volatility.loc[session].fillna(0)
            liquidity = adv.loc[session].fillna(0)
            try:
                projected = estimate_costs(
                    active.to_numpy(), target.to_numpy(), vol.to_numpy(), liquidity.to_numpy(),
                    config.nav, PortfolioCosts(
                        commission_bps=costs.commission_bps,
                        half_spread_bps=costs.half_spread_bps,
                        slippage_bps=costs.slippage_bps,
                        impact_coefficient=costs.impact_coefficient,
                        borrow_annual=0.0,
                        calendar_days=1.0,
                        multiplier=costs.multiplier,
                    ),
                )
                pending_cost = projected["transaction"]
                pending_turnover = float((target - active).abs().sum())
                active = target
            except ValueError:
                pass
        last_session = session
    return pd.DataFrame(records).set_index("session"), pd.DataFrame(weights)


def performance_metrics(returns: pd.Series) -> dict[str, float | int]:
    usable = returns.dropna()
    if usable.empty:
        return {"sessions": 0}
    nav = (1 + usable).cumprod()
    years = len(usable) / 252
    volatility = usable.std(ddof=1) * np.sqrt(252)
    return {
        "sessions": len(usable),
        "total_return": float(nav.iloc[-1] - 1),
        "cagr": float(nav.iloc[-1] ** (1 / years) - 1),
        "annual_volatility": float(volatility),
        "sharpe": float(usable.mean() * 252 / volatility) if volatility > 0 else np.nan,
        "max_drawdown": float((nav / nav.cummax() - 1).min()),
        "positive_day_rate": float(usable.gt(0).mean()),
    }

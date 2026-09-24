"""Causal adaptive Kalman, orthogonal alpha blending, and cost-confidence filters."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class AdaptiveKalmanConfig:
    warmup: int = 60
    r_half_life: float = 20.0
    initial_r: float = 1e-4
    fixed_q: float = 1e-5
    adaptive_q_scale: float = 1e-4
    q_floor: float = 1e-7
    q_cap: float = 1e-2
    r_floor: float = 1e-8
    covariance_floor: float = 1e-8
    covariance_cap: float = 10.0

    def __post_init__(self):
        values = (
            self.r_half_life, self.initial_r, self.fixed_q, self.adaptive_q_scale,
            self.q_floor, self.q_cap, self.r_floor, self.covariance_floor,
            self.covariance_cap,
        )
        if not isinstance(self.warmup, int) or self.warmup < 1:
            raise ValueError("warmup must be a positive integer")
        if not np.isfinite(values).all() or min(values) <= 0:
            raise ValueError("adaptive Kalman settings must be finite and positive")
        if self.q_floor > self.q_cap or self.covariance_floor > self.covariance_cap:
            raise ValueError("adaptive Kalman floors cannot exceed caps")


@dataclass(frozen=True)
class AdaptiveKalmanResult:
    signal: pd.DataFrame
    innovation: pd.DataFrame
    beta_prior: pd.DataFrame
    q_prior: pd.DataFrame
    r_prior: pd.DataFrame


@dataclass(frozen=True)
class BlendArtifact:
    families: tuple[str, ...]
    weights: tuple[float, ...]
    mean_rank_ic: tuple[float, ...]
    icir: tuple[float, ...]
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    digest: str


@dataclass(frozen=True)
class CalibrationArtifact:
    slope: float
    penalty: float
    observations: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    digest: str


def adaptive_kalman_innovation(
    returns: pd.DataFrame,
    market: pd.Series,
    *,
    mode: str = "qr_adaptive",
    config: AdaptiveKalmanConfig | None = None,
) -> AdaptiveKalmanResult:
    """Generate innovations with Q/R known before each session's observation."""
    config = config or AdaptiveKalmanConfig()
    if mode not in {"r_adaptive", "qr_adaptive"}:
        raise ValueError("mode must be r_adaptive or qr_adaptive")
    if not returns.index.equals(market.index):
        raise ValueError("returns and market sessions must align")
    values = returns.to_numpy(dtype=float)
    market_values = market.to_numpy(dtype=float)
    n_sessions, n_names = values.shape
    beta = np.ones(n_names)
    covariance = np.ones(n_names)
    measurement_variance = np.full(n_names, config.initial_r)
    counts = np.zeros(n_names, dtype=int)
    beta_history = np.full_like(values, np.nan)
    innovation_history = np.full_like(values, np.nan)
    signal_history = np.full_like(values, np.nan)
    q_history = np.full_like(values, np.nan)
    r_history = np.full_like(values, np.nan)
    market_variance = market.rolling(60, min_periods=60).var().shift(1).to_numpy()
    vol20 = market.rolling(20, min_periods=20).std().shift(1).to_numpy()
    vol60 = market.rolling(60, min_periods=60).std().shift(1).to_numpy()
    alpha = 1 - np.exp(np.log(0.5) / config.r_half_life)

    for offset in range(n_sessions):
        x = market_values[offset]
        if mode == "r_adaptive" or not np.isfinite(market_variance[offset]):
            process_variance = np.full(n_names, config.fixed_q)
        else:
            denominator = max(float(market_variance[offset]), config.r_floor)
            regime = (
                np.clip(vol20[offset] / vol60[offset], 0.5, 2.0)
                if np.isfinite(vol20[offset]) and np.isfinite(vol60[offset])
                and vol60[offset] > 0
                else 1.0
            )
            process_variance = np.clip(
                config.adaptive_q_scale * measurement_variance / denominator * regime,
                config.q_floor,
                config.q_cap,
            )
        covariance = np.clip(
            covariance + process_variance, config.covariance_floor, config.covariance_cap
        )
        beta_history[offset] = beta
        q_history[offset] = process_variance
        r_history[offset] = measurement_variance
        y = values[offset]
        usable = np.isfinite(y) & np.isfinite(x)
        if not usable.any():
            continue
        error = y - beta * x
        innovation_variance = measurement_variance + covariance * x * x
        ready = usable & (counts >= config.warmup) & (innovation_variance > 0)
        innovation_history[offset, ready] = error[ready]
        signal_history[offset, ready] = -error[ready] / np.sqrt(innovation_variance[ready])
        gain = covariance * x / innovation_variance
        beta[usable] += gain[usable] * error[usable]
        covariance[usable] *= 1 - gain[usable] * x
        covariance = np.clip(
            covariance, config.covariance_floor, config.covariance_cap
        )
        measurement_variance[usable] = np.maximum(
            (1 - alpha) * measurement_variance[usable] + alpha * error[usable] ** 2,
            config.r_floor,
        )
        counts[usable] += 1

    def frame(array: np.ndarray) -> pd.DataFrame:
        return pd.DataFrame(array, index=returns.index, columns=returns.columns)

    return AdaptiveKalmanResult(
        signal=frame(signal_history),
        innovation=frame(innovation_history),
        beta_prior=frame(beta_history),
        q_prior=frame(q_history),
        r_prior=frame(r_history),
    )


def remove_sector_peer_mean(
    signal: pd.DataFrame, sectors: pd.Series, eligibility: pd.DataFrame
) -> pd.DataFrame:
    if (
        not signal.columns.equals(sectors.index)
        or not signal.index.equals(eligibility.index)
        or not signal.columns.equals(eligibility.columns)
    ):
        raise ValueError("signal, sectors, and eligibility must align")
    output = signal.copy() * np.nan
    for sector in sorted(sectors.dropna().unique()):
        names = sectors.index[sectors.eq(sector)]
        group = signal[names].where(eligibility[names])
        counts = group.notna().sum(axis=1)
        peer_mean = group.rsub(group.sum(axis=1, min_count=1), axis=0).div(
            (counts - 1).where(counts >= 3), axis=0
        )
        output[names] = group - peer_mean
    return output


def _zscore(frame: pd.DataFrame, eligibility: pd.DataFrame) -> pd.DataFrame:
    usable = frame.where(eligibility)
    centered = usable.sub(usable.mean(axis=1), axis=0)
    scale = usable.std(axis=1, ddof=1).where(usable.count(axis=1).ge(2))
    return centered.div(scale.where(scale.gt(0)), axis=0).clip(-3, 3)


def orthogonalize_families(
    families: dict[str, pd.DataFrame], eligibility: pd.DataFrame, minimum_names: int = 50
) -> dict[str, pd.DataFrame]:
    """Daily cross-sectional Gram-Schmidt in caller-provided family order."""
    if not families:
        raise ValueError("at least one family is required")
    first = next(iter(families.values()))
    if not first.index.equals(eligibility.index) or not first.columns.equals(eligibility.columns):
        raise ValueError("eligibility must align with families")
    if any(
        not frame.index.equals(first.index) or not frame.columns.equals(first.columns)
        for frame in families.values()
    ):
        raise ValueError("all family frames must align")
    outputs = {name: first.copy() * np.nan for name in families}
    for session in first.index:
        raw = pd.DataFrame(
            {name: frame.loc[session] for name, frame in families.items()}, index=first.columns
        )
        usable = eligibility.loc[session] & raw.notna().all(axis=1)
        if usable.sum() < minimum_names:
            continue
        prior: list[np.ndarray] = []
        for name in families:
            vector = raw.loc[usable, name].to_numpy(dtype=float)
            design = np.column_stack([np.ones(len(vector)), *prior])
            residual = vector - design @ np.linalg.lstsq(design, vector, rcond=None)[0]
            scale = residual.std(ddof=1)
            if not np.isfinite(scale) or scale <= 0:
                break
            standardized = np.clip((residual - residual.mean()) / scale, -3, 3)
            outputs[name].loc[session, usable] = standardized
            prior.append(standardized)
    return outputs


def fit_icir_blend(
    orthogonal: dict[str, pd.DataFrame],
    labels: pd.DataFrame,
    label_end: pd.DataFrame,
    train_start,
    train_end,
) -> BlendArtifact:
    start, end = pd.Timestamp(train_start), pd.Timestamp(train_end)
    families = tuple(orthogonal)
    mean_ics, icirs, raw_weights = [], [], []
    for name in families:
        completed = label_end.le(end)
        signal = orthogonal[name].where(completed)
        target = labels.where(completed)
        usable = signal.corrwith(target, axis=1, method="spearman").loc[start:end]
        mean_ic = float(usable.mean())
        dispersion = float(usable.std(ddof=1))
        icir = mean_ic / dispersion if dispersion > 0 else np.nan
        mean_ics.append(mean_ic)
        icirs.append(icir)
        raw_weights.append(max(icir, 0) if np.isfinite(icir) else 0.0)
    if sum(weight > 0 for weight in raw_weights) < 2:
        raise ValueError("FEWER_THAN_TWO_POSITIVE_TRAIN_ICIR_FAMILIES")
    weights = np.asarray(raw_weights) / sum(raw_weights)
    payload = np.r_[mean_ics, icirs, weights].astype(float).tobytes()
    return BlendArtifact(
        families=families,
        weights=tuple(float(value) for value in weights),
        mean_rank_ic=tuple(mean_ics),
        icir=tuple(icirs),
        train_start=start,
        train_end=end,
        digest=hashlib.sha256(payload).hexdigest(),
    )


def blend_families(
    orthogonal: dict[str, pd.DataFrame], weights: dict[str, float]
) -> pd.DataFrame:
    if set(weights) != set(orthogonal):
        raise ValueError("blend weights must name every orthogonal family")
    if not np.isclose(sum(weights.values()), 1) or min(weights.values()) < 0:
        raise ValueError("blend weights must be nonnegative and sum to one")
    return sum(weights[name] * orthogonal[name] for name in orthogonal)


def fit_zero_intercept_calibration(
    score: pd.DataFrame,
    labels: pd.DataFrame,
    label_end: pd.DataFrame,
    train_start,
    train_end,
    penalty: float = 1e-6,
) -> CalibrationArtifact:
    start, end = pd.Timestamp(train_start), pd.Timestamp(train_end)
    completed = label_end.le(end)
    centered_labels = labels.sub(labels.mean(axis=1), axis=0)
    x = score.loc[start:end].where(completed.loc[start:end]).to_numpy().ravel()
    y = centered_labels.loc[start:end].where(completed.loc[start:end]).to_numpy().ravel()
    usable = np.isfinite(x) & np.isfinite(y)
    if usable.sum() < 100_000:
        raise ValueError("INSUFFICIENT_CALIBRATION_OBSERVATIONS")
    numerator = float(x[usable] @ y[usable])
    denominator = float(x[usable] @ x[usable] + penalty)
    slope = numerator / denominator
    if not np.isfinite(slope) or slope <= 0:
        raise ValueError("NONPOSITIVE_TRAIN_CALIBRATION_SLOPE")
    digest = hashlib.sha256(np.c_[x[usable], y[usable]].astype(float).tobytes()).hexdigest()
    return CalibrationArtifact(slope, penalty, int(usable.sum()), start, end, digest)


def expected_return(score: pd.DataFrame, artifact: CalibrationArtifact) -> pd.DataFrame:
    return score * artifact.slope


def cost_hurdle(
    forecast: pd.DataFrame,
    prior_daily_volatility: pd.DataFrame,
    prior_adv: pd.DataFrame,
    *,
    nav: float = 100_000.0,
    assumed_one_way_weight: float = 0.005,
) -> pd.DataFrame:
    if not (
        forecast.index.equals(prior_daily_volatility.index)
        and forecast.columns.equals(prior_daily_volatility.columns)
        and forecast.index.equals(prior_adv.index)
        and forecast.columns.equals(prior_adv.columns)
    ):
        raise ValueError("forecast, volatility, and ADV must align")
    base = 2 * (0.5 + 2.0 + 1.0) / 10_000
    participation = assumed_one_way_weight * nav / prior_adv.where(prior_adv.gt(0))
    impact = 2 * 0.10 * prior_daily_volatility * np.sqrt(participation)
    borrow = forecast.lt(0).astype(float) * 0.03 * 3 / 365
    return base + impact + borrow


def apply_confidence_filter(
    score: pd.DataFrame,
    forecast: pd.DataFrame,
    hurdle: pd.DataFrame,
    multiplier: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if multiplier <= 0 or not np.isfinite(multiplier):
        raise ValueError("confidence multiplier must be finite and positive")
    if not (
        score.index.equals(forecast.index)
        and score.columns.equals(forecast.columns)
        and score.index.equals(hurdle.index)
        and score.columns.equals(hurdle.columns)
    ):
        raise ValueError("score, forecast, and hurdle must align")
    passed = forecast.abs().ge(multiplier * hurdle) & hurdle.notna()
    return score.where(passed), passed

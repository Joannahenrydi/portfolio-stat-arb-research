"""Causal daily equity signals and train-only calibration primitives.

All wide frames use permanent security IDs as columns and the same explicit market
session index. Missing sessions/returns stay missing: callers must reindex to their
verified exchange calendar before using these functions. Sector labels and
eligibility are session-specific inputs, never today's labels projected backward.

These are research primitives, not evidence that any supplied history is PIT.
Calibration requires either a genuine PIT attestation or an explicit
research_snapshot_only provenance label. The latter permits current-cohort
research but never claims historical membership validity.
Signals at session t use t's completed observations with parameters estimated
strictly before t. Orders based on them belong to a later executable time.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd


class AlphaDataError(ValueError):
    """Data cannot support the requested causal estimate."""


@dataclass(frozen=True)
class AlphaConfig:
    market_window: int = 126
    sector_window: int = 126
    reversal_window: int = 3
    volatility_window: int = 60
    volume_window: int = 60
    min_sector_peers: int = 2
    kalman_min_observations: int = 60
    kalman_process_variance: float = 1e-5
    kalman_measurement_variance: float = 1e-4
    max_volume_surprise: float = 5.0
    zscore_clip: float = 3.0

    def __post_init__(self):
        for name in (
            "market_window",
            "sector_window",
            "reversal_window",
            "volatility_window",
            "volume_window",
            "min_sector_peers",
            "kalman_min_observations",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if min(self.market_window, self.sector_window, self.volatility_window) < 2:
            raise ValueError("regression/volatility windows require at least two sessions")
        positive = [self.kalman_measurement_variance, self.max_volume_surprise, self.zscore_clip]
        if not np.isfinite(positive).all() or min(positive) <= 0:
            raise ValueError("measurement variance and signal clips must be finite and positive")
        if not np.isfinite(self.kalman_process_variance) or self.kalman_process_variance < 0:
            raise ValueError("Kalman process variance must be finite and nonnegative")


@dataclass(frozen=True)
class EquityFeatures:
    signals: Mapping[str, pd.DataFrame]
    market_beta_prior: pd.DataFrame
    market_residual: pd.DataFrame
    sector_beta_prior: pd.DataFrame
    sector_factor: pd.DataFrame
    residual: pd.DataFrame
    kalman_beta_prior: pd.DataFrame
    kalman_innovation: pd.DataFrame
    volatility_prior: pd.DataFrame
    volume_surprise: pd.DataFrame


@dataclass(frozen=True)
class CalibrationArtifact:
    """Frozen training sufficient statistics; prediction never refits them."""

    families: tuple[str, ...]
    coefficients: tuple[float, ...]
    feature_means: tuple[float, ...]
    feature_scales: tuple[float, ...]
    intercept: float
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    latest_label_session: pd.Timestamp
    sample_count: int
    ridge: float
    training_digest: str
    pit_verified: bool
    input_scope: str


@dataclass(frozen=True)
class CovarianceEstimate:
    covariance: pd.DataFrame
    decision_session: pd.Timestamp
    window_start: pd.Timestamp
    window_end: pd.Timestamp
    observations: pd.Series
    pairwise_observations: pd.DataFrame
    shrinkage: float
    eigenvalue_floor: float
    repaired_eigenvalues: int


def _session(value, name: str) -> pd.Timestamp:
    value = pd.Timestamp(value)
    if pd.isna(value) or value.tzinfo is not None or value != value.normalize():
        raise AlphaDataError(f"{name} must be a timezone-free market-session date")
    return value


def _frame(frame: pd.DataFrame, name: str, numeric: bool = True) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame) or not isinstance(frame.index, pd.DatetimeIndex):
        raise AlphaDataError(f"{name} needs a DataFrame with a DatetimeIndex")
    if frame.empty or frame.index.has_duplicates or not frame.index.is_monotonic_increasing:
        raise AlphaDataError(f"{name} needs nonempty, unique, increasing sessions")
    if frame.index.tz is not None or not frame.index.equals(frame.index.normalize()):
        raise AlphaDataError(f"{name} index must contain timezone-free market-session dates")
    if frame.columns.has_duplicates or not all(
        isinstance(column, str) and column.strip() for column in frame.columns
    ):
        raise AlphaDataError(f"{name} columns must be unique permanent security IDs")
    frame = frame.copy()
    if numeric:
        try:
            frame = frame.astype(float)
        except (TypeError, ValueError) as exc:
            raise AlphaDataError(f"{name} must be numeric") from exc
        if np.isinf(frame.to_numpy()).any():
            raise AlphaDataError(f"{name} cannot contain infinite values")
    return frame


def _aligned(frame: pd.DataFrame, reference: pd.DataFrame, name: str) -> None:
    if not frame.index.equals(reference.index) or not frame.columns.equals(reference.columns):
        raise AlphaDataError(f"{name} must have exactly the same sessions and security IDs")


def _zscore(frame: pd.DataFrame, eligibility: pd.DataFrame, clip: float) -> pd.DataFrame:
    usable = frame.where(eligibility)
    centered = usable.sub(usable.mean(axis=1), axis=0)
    std = usable.std(axis=1, ddof=1).where(usable.count(axis=1) >= 2)
    return centered.div(std.where(std > 0), axis=0).clip(-clip, clip)


def _sector_peer_factor(
    values: pd.DataFrame,
    sectors: pd.DataFrame,
    eligibility: pd.DataFrame,
    min_peers: int,
) -> pd.DataFrame:
    """Contemporaneous leave-one-out means avoid a stock explaining its own return."""
    output = pd.DataFrame(np.nan, index=values.index, columns=values.columns)
    labels = sorted(set(sectors.stack().dropna()))
    for label in labels:
        group = values.where(sectors.eq(label) & eligibility)
        count = group.count(axis=1)
        sums = group.sum(axis=1, min_count=1)
        peer_sum = group.rsub(sums, axis=0)
        peer_count = count - 1
        factor = peer_sum.div(peer_count.where(peer_count >= min_peers), axis=0)
        output = output.mask(sectors.eq(label), factor)
    return output


def _kalman_prior(
    returns: pd.DataFrame,
    market: pd.Series,
    config: AlphaConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    beta = np.ones(len(returns.columns))
    uncertainty = np.ones(len(returns.columns))
    counts = np.zeros(len(returns.columns), dtype=int)
    prior = np.full(returns.shape, np.nan)
    innovations = np.full(returns.shape, np.nan)
    for t, (x, y) in enumerate(zip(market.to_numpy(), returns.to_numpy())):
        uncertainty += config.kalman_process_variance
        prior[t] = beta
        if not np.isfinite(x):
            continue
        usable = np.isfinite(y)
        error = y - beta * x  # Must be formed before the state sees y_t.
        ready = usable & (counts >= config.kalman_min_observations)
        innovations[t, ready] = error[ready]
        gain = uncertainty * x / (config.kalman_measurement_variance + uncertainty * x * x)
        beta[usable] += gain[usable] * error[usable]
        uncertainty[usable] *= 1 - gain[usable] * x
        counts[usable] += 1
    return (
        pd.DataFrame(prior, index=returns.index, columns=returns.columns),
        pd.DataFrame(innovations, index=returns.index, columns=returns.columns),
    )


def build_equity_features(
    returns: pd.DataFrame,
    market_returns: pd.Series,
    volume: pd.DataFrame,
    sectors: pd.DataFrame,
    eligibility: pd.DataFrame,
    config: AlphaConfig | None = None,
) -> EquityFeatures:
    """Construct causal reversal, prior-Kalman, and volume/volatility signals.

    ``returns`` should be verified total returns, including terminal outcomes.
    ``sectors`` and ``eligibility`` must be PIT observations for each session.
    No current membership classification is inferred by this function.
    """
    config = config or AlphaConfig()
    returns = _frame(returns, "returns")
    volume = _frame(volume, "volume")
    sectors = _frame(sectors, "sectors", numeric=False)
    eligibility = _frame(eligibility, "eligibility", numeric=False)
    for name, frame in (("volume", volume), ("sectors", sectors), ("eligibility", eligibility)):
        _aligned(frame, returns, name)
    if not isinstance(market_returns, pd.Series) or not market_returns.index.equals(returns.index):
        raise AlphaDataError("market_returns must have exactly the same market sessions")
    market = market_returns.astype(float)
    if np.isinf(market.to_numpy()).any() or (market < -1).any() or (returns < -1).any().any():
        raise AlphaDataError("simple total returns must be finite or missing and at least -1")
    if (volume < 0).any().any():
        raise AlphaDataError("volume cannot be negative")
    if (
        not eligibility.apply(lambda column: column.map(lambda x: isinstance(x, (bool, np.bool_))))
        .to_numpy()
        .all()
    ):
        raise AlphaDataError("eligibility must contain explicit booleans")
    eligibility = eligibility.astype(bool)
    sector_values = sectors.stack().dropna()
    if not sector_values.map(lambda x: isinstance(x, str) and bool(x.strip())).all():
        raise AlphaDataError("sector labels must be nonempty strings or missing")
    eligible = eligibility & sectors.notna()
    mw = config.market_window
    market_variance = market.rolling(mw, min_periods=mw).var().shift(1)
    beta = (
        returns.rolling(mw, min_periods=mw)
        .cov(market)
        .shift(1)
        .div(market_variance.where(market_variance > 0), axis=0)
    )
    rmean = returns.rolling(mw, min_periods=mw).mean().shift(1)
    mmean = market.rolling(mw, min_periods=mw).mean().shift(1)
    market_residual = returns - rmean - beta.mul(market - mmean, axis=0)
    sector_factor = _sector_peer_factor(market_residual, sectors, eligible, config.min_sector_peers)
    sw = config.sector_window
    sector_variance = sector_factor.rolling(sw, min_periods=sw).var().shift(1)
    sector_beta = (
        market_residual.rolling(sw, min_periods=sw)
        .cov(sector_factor)
        .shift(1)
        .div(sector_variance.where(sector_variance > 0))
    )
    sector_mean = sector_factor.rolling(sw, min_periods=sw).mean().shift(1)
    residual_mean = market_residual.rolling(sw, min_periods=sw).mean().shift(1)
    residual = market_residual - residual_mean - sector_beta * (sector_factor - sector_mean)
    kalman_beta, innovations = _kalman_prior(returns, market, config)
    kalman_sector = _sector_peer_factor(innovations, sectors, eligible, config.min_sector_peers)
    kalman_residual = innovations - kalman_sector
    vol = (
        returns.rolling(config.volatility_window, min_periods=config.volatility_window)
        .std()
        .shift(1)
    )
    normal_volume = (
        volume.rolling(config.volume_window, min_periods=config.volume_window).median().shift(1)
    )
    surprise = volume.div(normal_volume.where(normal_volume > 0)).clip(
        0, config.max_volume_surprise
    )
    reversal = -residual.rolling(config.reversal_window, min_periods=config.reversal_window).sum()
    signals = {
        "residual_reversal": _zscore(reversal, eligible, config.zscore_clip),
        "kalman_innovation": _zscore(-kalman_residual, eligible, config.zscore_clip),
        "volume_volatility_dislocation": _zscore(
            -residual * surprise / vol.where(vol > 0), eligible, config.zscore_clip
        ),
    }
    return EquityFeatures(
        signals,
        beta,
        market_residual,
        sector_beta,
        sector_factor,
        residual,
        kalman_beta,
        innovations,
        vol,
        surprise,
    )


def forward_total_return_labels(
    returns: pd.DataFrame,
    horizon: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Next-h-session compounded labels and the session when each label completes.

    This constructs outcomes for training, never predictors. Missing any component
    return makes that label missing. The final h rows have no completed label.
    """
    returns = _frame(returns, "returns")
    if not isinstance(horizon, int) or horizon < 1:
        raise ValueError("horizon must be a positive integer")
    if (returns < -1).any().any():
        raise AlphaDataError("simple total returns cannot be less than -1")
    gross = pd.DataFrame(1.0, index=returns.index, columns=returns.columns)
    for step in range(1, horizon + 1):
        gross *= 1 + returns.shift(-step)
    end = pd.Series(returns.index, index=returns.index).shift(-horizon)
    label_end = pd.DataFrame({column: end for column in returns.columns})
    return gross - 1, label_end


def _signal_frames(signals: Mapping[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    if not signals or not all(isinstance(name, str) and name for name in signals):
        raise AlphaDataError("at least one named signal is required")
    frames = {name: _frame(signals[name], f"signal {name}") for name in sorted(signals)}
    first = next(iter(frames.values()))
    for name, frame in frames.items():
        _aligned(frame, first, f"signal {name}")
    return frames


def fit_expected_return_calibration(
    signals: Mapping[str, pd.DataFrame],
    forward_returns: pd.DataFrame,
    label_end: pd.DataFrame,
    *,
    train_start,
    train_end,
    pit_verified: bool,
    input_scope: str = "historical_pit",
    ridge: float = 0.1,
    min_samples: int = 500,
) -> CalibrationArtifact:
    """Fit pooled ridge calibration solely inside the explicit training segment.

    Only rows with train_start <= signal session <= label completion <= train_end
    enter the artifact. Validation/test labels cannot cross the training boundary.
    ``pit_verified=True`` is an evidence attestation by the caller. For
    current-cohort research, explicitly set input_scope="research_snapshot_only"
    and pit_verified=False; the artifact preserves that limitation.
    """
    if input_scope not in {"historical_pit", "research_snapshot_only"}:
        raise AlphaDataError("input_scope must explicitly identify PIT or snapshot research")
    if input_scope == "historical_pit" and pit_verified is not True:
        raise AlphaDataError("historical_pit calibration requires genuinely verified PIT inputs")
    if input_scope == "research_snapshot_only" and pit_verified is not False:
        raise AlphaDataError("snapshot research must preserve pit_verified=False")
    if not np.isfinite(ridge) or ridge < 0:
        raise ValueError("ridge must be finite and nonnegative")
    if not isinstance(min_samples, int) or min_samples < 2:
        raise ValueError("min_samples must be an integer >= 2")
    frames = _signal_frames(signals)
    reference = next(iter(frames.values()))
    outcomes = _frame(forward_returns, "forward_returns")
    ends = _frame(label_end, "label_end", numeric=False)
    _aligned(outcomes, reference, "forward_returns")
    _aligned(ends, reference, "label_end")
    start, end = _session(train_start, "train_start"), _session(train_end, "train_end")
    if start > end:
        raise AlphaDataError("train_start cannot follow train_end")
    mask = (reference.index >= start) & (reference.index <= end)
    if not mask.any():
        raise AlphaDataError("training segment has no signal sessions")
    families = tuple(frames)
    x = np.stack([frames[name].loc[mask].to_numpy() for name in families], axis=-1)
    y = outcomes.loc[mask].to_numpy()
    try:
        end_values = ends.loc[mask].apply(pd.to_datetime).to_numpy(dtype="datetime64[ns]")
    except (TypeError, ValueError) as exc:
        raise AlphaDataError("label_end must contain session dates or NaT") from exc
    observed_ends = end_values[~np.isnat(end_values)]
    if observed_ends.size and not np.all(
        observed_ends.astype("datetime64[D]").astype("datetime64[ns]") == observed_ends
    ):
        raise AlphaDataError("label_end must contain normalized session dates")
    signal_sessions = reference.index[mask].to_numpy()[:, None]
    complete = (
        ~np.isnat(end_values) & (end_values > signal_sessions) & (end_values <= np.datetime64(end))
    )
    usable = np.isfinite(x).all(axis=2) & np.isfinite(y) & complete
    x, y = x[usable], y[usable]
    if len(y) < min_samples:
        raise AlphaDataError(f"only {len(y)} completed training samples; {min_samples} required")
    means, scales = x.mean(axis=0), x.std(axis=0)
    if (scales <= 1e-12).any():
        raise AlphaDataError("training signal is degenerate; cannot calibrate")
    normalized = (x - means) / scales
    intercept = float(y.mean())
    gram = normalized.T @ normalized / len(y) + ridge * np.eye(len(families))
    rhs = normalized.T @ (y - intercept) / len(y)
    coefficients = np.linalg.lstsq(gram, rhs, rcond=None)[0]
    digest = hashlib.sha256()
    digest.update(repr((families, str(start), str(end), ridge, input_scope)).encode())
    digest.update(np.asarray(x, dtype="<f8").tobytes())
    digest.update(np.asarray(y, dtype="<f8").tobytes())
    digest.update(end_values[usable].astype("datetime64[ns]").astype("<i8").tobytes())
    return CalibrationArtifact(
        families,
        tuple(coefficients),
        tuple(means),
        tuple(scales),
        intercept,
        start,
        end,
        pd.Timestamp(end_values[usable].max()),
        len(y),
        ridge,
        digest.hexdigest(),
        pit_verified,
        input_scope,
    )


def predict_expected_returns(
    signals: Mapping[str, pd.DataFrame],
    artifact: CalibrationArtifact,
) -> pd.DataFrame:
    """Apply one frozen artifact strictly after its training segment ends."""
    if artifact.input_scope == "historical_pit" and not artifact.pit_verified:
        raise AlphaDataError("PIT calibration artifact lacks verified PIT inputs")
    if artifact.input_scope not in {"historical_pit", "research_snapshot_only"}:
        raise AlphaDataError("calibration artifact lacks an explicit input scope")
    frames = _signal_frames(signals)
    if tuple(frames) != artifact.families:
        raise AlphaDataError("prediction signal families differ from frozen training artifact")
    first = next(iter(frames.values()))
    if (first.index <= artifact.train_end).any():
        raise AlphaDataError("predictions must be strictly after the training segment")
    values = np.stack([frames[name].to_numpy() for name in artifact.families], axis=-1)
    predictions = artifact.intercept + (
        (values - np.asarray(artifact.feature_means)) / np.asarray(artifact.feature_scales)
    ) @ np.asarray(artifact.coefficients)
    predictions[~np.isfinite(values).all(axis=2)] = np.nan
    return pd.DataFrame(predictions, index=first.index, columns=first.columns)


def estimate_shrinkage_covariance(
    returns: pd.DataFrame,
    decision_session,
    *,
    lookback: int = 252,
    min_observations: int = 126,
    shrinkage: float = 0.2,
    eigenvalue_floor_ratio: float = 1e-8,
) -> CovarianceEstimate:
    """Daily covariance from sessions strictly before decision_session.

    Missing returns use pairwise-complete observations with an explicit minimum
    for every pair. Diagonal shrinkage plus a reported eigenvalue repair yields a
    positive definite matrix without filling missing returns or assuming zero.
    """
    returns = _frame(returns, "returns")
    decision = _session(decision_session, "decision_session")
    if not isinstance(lookback, int) or not isinstance(min_observations, int):
        raise TypeError("covariance windows must be integers")
    if not 2 <= min_observations <= lookback:
        raise ValueError("require 2 <= min_observations <= lookback")
    if not np.isfinite(shrinkage) or not 0 <= shrinkage <= 1:
        raise ValueError("shrinkage must lie in [0, 1]")
    if not np.isfinite(eigenvalue_floor_ratio) or eigenvalue_floor_ratio <= 0:
        raise ValueError("eigenvalue_floor_ratio must be finite and positive")
    past = returns.loc[returns.index < decision].tail(lookback)
    if len(past) < min_observations:
        raise AlphaDataError("insufficient prior sessions for covariance")
    present = past.notna().astype(np.int64)
    pairwise = present.T @ present
    if (pairwise < min_observations).any().any():
        raise AlphaDataError("insufficient pairwise prior observations for covariance")
    sample = past.cov(min_periods=min_observations).to_numpy()
    diagonal = np.diag(sample)
    if not np.isfinite(sample).all() or (diagonal <= 0).any():
        raise AlphaDataError("covariance requires finite, positive asset variances")
    matrix = (1 - shrinkage) * sample + shrinkage * np.diag(diagonal)
    matrix = (matrix + matrix.T) / 2
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    floor = float(diagonal.max() * eigenvalue_floor_ratio)
    repaired = int((eigenvalues < floor).sum())
    if repaired:
        matrix = (eigenvectors * np.maximum(eigenvalues, floor)) @ eigenvectors.T
        matrix = (matrix + matrix.T) / 2
    covariance = pd.DataFrame(matrix, index=returns.columns, columns=returns.columns)
    return CovarianceEstimate(
        covariance,
        decision,
        past.index[0],
        past.index[-1],
        present.sum(),
        pairwise,
        shrinkage,
        floor,
        repaired,
    )

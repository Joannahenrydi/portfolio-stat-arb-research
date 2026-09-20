from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class StrategyConfig:
    formation_window: int = 126
    z_window: int = 60
    entry_z: float = 1.0
    exit_z: float = 0.15
    stop_z: float = 3.5
    max_holding_bars: int = 60
    process_variance: float = 1e-5
    measurement_variance: float | None = None
    initial_capital: float = 100_000.0
    gross_pair_fraction: float = 0.20
    commission_bps: float = 1.0
    slippage_bps: float = 2.0
    max_data_age_hours: int = 96

    def __post_init__(self) -> None:
        if self.formation_window < 30 or self.z_window < 20:
            raise ValueError("formation_window >= 30 and z_window >= 20 are required")
        if not (0 < self.exit_z < self.entry_z < self.stop_z):
            raise ValueError("require 0 < exit_z < entry_z < stop_z")
        if not (0 < self.gross_pair_fraction <= 1):
            raise ValueError("gross_pair_fraction must be in (0, 1]")


class KalmanHedgeRatio:
    """Scalar random-walk Kalman filter: y_t = beta_t*x_t + epsilon_t."""

    def __init__(self, beta: float, process_variance: float, measurement_variance: float):
        self.beta = float(beta)
        self.p = 1.0
        self.q = float(process_variance)
        self.r = max(float(measurement_variance), 1e-12)

    def update(self, x: float, y: float) -> tuple[float, float]:
        p_prior = self.p + self.q
        innovation = y - self.beta * x
        innovation_var = x * x * p_prior + self.r
        gain = p_prior * x / innovation_var
        predicted_beta = self.beta
        self.beta = predicted_beta + gain * innovation
        self.p = max((1.0 - gain * x) * p_prior, 1e-12)
        return predicted_beta, innovation


def build_signals(prices: pd.DataFrame, x: str, y: str, cfg: StrategyConfig) -> pd.DataFrame:
    frame = prices[[x, y]].astype(float).dropna().sort_index()
    if len(frame) < cfg.formation_window + cfg.z_window + 2:
        raise ValueError("not enough observations for formation and signal windows")
    if (frame <= 0).any().any() or frame.index.has_duplicates:
        raise ValueError("prices must be positive and dates unique")

    formation = frame.iloc[: cfg.formation_window]
    beta0 = float(np.dot(formation[x], formation[y]) / np.dot(formation[x], formation[x]))
    formation_spread = formation[y] - beta0 * formation[x]
    measurement_var = cfg.measurement_variance or float(formation_spread.var(ddof=1))
    kf = KalmanHedgeRatio(beta0, cfg.process_variance, measurement_var)

    rows: list[dict[str, float]] = []
    for i, (idx, row) in enumerate(frame.iterrows()):
        if i < cfg.formation_window:
            beta_prior = beta0
            innovation = float(row[y]) - beta0 * float(row[x])
        else:
            beta_prior, innovation = kf.update(float(row[x]), float(row[y]))
        rows.append({"date": idx, "beta": beta_prior, "spread": innovation})
    result = pd.DataFrame(rows).set_index("date")
    # Shift makes today's z-score use spread history available before today's close.
    history = result["spread"].shift(1)
    mean = history.rolling(cfg.z_window, min_periods=cfg.z_window).mean()
    std = history.rolling(cfg.z_window, min_periods=cfg.z_window).std(ddof=1)
    result["zscore"] = (result["spread"] - mean) / std.replace(0.0, np.nan)
    result.loc[result.index[: cfg.formation_window], "zscore"] = np.nan
    return frame.join(result)

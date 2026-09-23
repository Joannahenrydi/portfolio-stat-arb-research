"""Expanding walk-forward expected-return calibration."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import pandas as pd

from features.alphas import (
    CalibrationArtifact,
    fit_expected_return_calibration,
    predict_expected_returns,
)


@dataclass(frozen=True)
class WalkForwardResult:
    predictions: pd.DataFrame
    artifacts: tuple[CalibrationArtifact, ...]


def expanding_walk_forward(
    signals: Mapping[str, pd.DataFrame],
    forward_returns: pd.DataFrame,
    label_end: pd.DataFrame,
    *,
    first_prediction,
    refit_every: int = 21,
    pit_verified: bool,
    input_scope: str,
    ridge: float = 0.1,
    min_samples: int = 500,
) -> WalkForwardResult:
    """Refit on completed past labels, then predict the next fixed session block.

    Each artifact ends on the session immediately before its prediction block.
    Labels that complete after that boundary are excluded by the calibration API.
    """
    if not signals or not isinstance(refit_every, int) or refit_every < 1:
        raise ValueError("signals are required and refit_every must be positive")
    first_frame = next(iter(signals.values()))
    sessions = first_frame.index
    first_prediction = pd.Timestamp(first_prediction)
    if first_prediction not in sessions:
        raise ValueError("first_prediction must be a signal session")
    start = sessions.get_loc(first_prediction)
    if start < 1:
        raise ValueError("at least one prior training session is required")
    prediction = pd.DataFrame(index=sessions[start:], columns=first_frame.columns, dtype=float)
    artifacts: list[CalibrationArtifact] = []
    for offset in range(start, len(sessions), refit_every):
        train_end = sessions[offset - 1]
        artifact = fit_expected_return_calibration(
            signals,
            forward_returns,
            label_end,
            train_start=sessions[0],
            train_end=train_end,
            pit_verified=pit_verified,
            input_scope=input_scope,
            ridge=ridge,
            min_samples=min_samples,
        )
        block_index = sessions[offset : offset + refit_every]
        block = {name: frame.loc[block_index] for name, frame in signals.items()}
        prediction.loc[block_index] = predict_expected_returns(block, artifact)
        artifacts.append(artifact)
    return WalkForwardResult(prediction, tuple(artifacts))

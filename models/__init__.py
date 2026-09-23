"""Expected-return calibration and covariance models."""

from .calibration import (
    CalibrationArtifact,
    CovarianceEstimate,
    estimate_shrinkage_covariance,
    fit_expected_return_calibration,
    predict_expected_returns,
)

__all__ = [
    "CalibrationArtifact",
    "CovarianceEstimate",
    "estimate_shrinkage_covariance",
    "fit_expected_return_calibration",
    "predict_expected_returns",
]
